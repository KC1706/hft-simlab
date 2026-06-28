//! Streaming reader for hftbacktest `.npz` event files.
//!
//! An `.npz` is a zip archive whose members are `.npy` arrays; ours holds a
//! single member `data.npy`: a 26M-row structured array of 64-byte `Event`
//! records, deflate-compressed (~197 MB on disk, ~1.7 GB raw). hftbacktest's
//! reader decompresses the whole array into memory; on an 8 GB machine we
//! stream instead — constant memory, one pass, records decoded on the fly.
//!
//! Container parsing is done by hand (it's ~80 lines and worth knowing):
//! - zip: the End Of Central Directory record (trailer) locates the central
//!   directory, which locates the member's local header and sizes. We read the
//!   trailer rather than trusting the local header because zip writers may
//!   leave local-header sizes zeroed (data-descriptor mode).
//! - npy: magic + version + a Python-dict header giving dtype and shape.
//!   We require the exact dtype our `Event` struct mirrors and fail loudly
//!   otherwise — silent layout drift would corrupt every downstream number.
//!
//! Format references: PKWARE APPNOTE (zip), numpy NEP "npy format" (npy),
//! refs/hftbacktest/hftbacktest/src/backtest/data/ (their reader, for parity).

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use flate2::bufread::DeflateDecoder;

use crate::events::Event;

/// The exact structured dtype written by hftbacktest's converters.
/// Field order and widths must mirror `Event` — checked at open time.
const EXPECTED_DESCR: &str = "[('ev', '<u8'), ('exch_ts', '<i8'), ('local_ts', '<i8'), \
('px', '<f8'), ('qty', '<f8'), ('order_id', '<u8'), ('ival', '<i8'), ('fval', '<f8')]";

const EVENT_SIZE: usize = 64;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn u16le(b: &[u8], off: usize) -> u64 {
    u16::from_le_bytes([b[off], b[off + 1]]) as u64
}

fn u32le(b: &[u8], off: usize) -> u64 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as u64
}

/// Location and size of the single compressed member inside the archive.
struct ZipMember {
    data_offset: u64,
    compressed_size: u64,
    raw_size: u64,
    method: u64, // 0 = stored, 8 = deflate
}

/// Locate the first member via EOCD -> central directory -> local header.
fn locate_member(f: &mut File) -> io::Result<ZipMember> {
    let file_len = f.seek(SeekFrom::End(0))?;

    // EOCD: 22-byte record + up to 64KB trailing comment; scan backward for its signature.
    let tail_len = file_len.min(22 + 65_536);
    f.seek(SeekFrom::Start(file_len - tail_len))?;
    let mut tail = vec![0u8; tail_len as usize];
    f.read_exact(&mut tail)?;
    let eocd = tail
        .windows(4)
        .rposition(|w| w == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| bad("not a zip: EOCD signature not found"))?;
    let eocd = &tail[eocd..];
    let n_entries = u16le(eocd, 10);
    let cd_offset = u32le(eocd, 16);
    if n_entries != 1 {
        return Err(bad(format!(
            "expected exactly 1 member in npz, found {n_entries}"
        )));
    }

    // Central directory entry (46-byte fixed part).
    f.seek(SeekFrom::Start(cd_offset))?;
    let mut cd = [0u8; 46];
    f.read_exact(&mut cd)?;
    if cd[..4] != [0x50, 0x4b, 0x01, 0x02] {
        return Err(bad("central directory signature mismatch"));
    }
    let method = u16le(&cd, 10);
    let compressed_size = u32le(&cd, 20);
    let raw_size = u32le(&cd, 24);
    let local_offset = u32le(&cd, 42);
    if compressed_size == u32::MAX as u64 || raw_size == u32::MAX as u64 {
        return Err(bad("zip64 archive — not supported (member >= 4 GiB)"));
    }
    if method != 0 && method != 8 {
        return Err(bad(format!("unsupported zip compression method {method}")));
    }

    // Local header (30-byte fixed part) carries its own name/extra lengths,
    // which may differ from the central directory's.
    f.seek(SeekFrom::Start(local_offset))?;
    let mut lh = [0u8; 30];
    f.read_exact(&mut lh)?;
    if lh[..4] != [0x50, 0x4b, 0x03, 0x04] {
        return Err(bad("local header signature mismatch"));
    }
    let name_len = u16le(&lh, 26);
    let extra_len = u16le(&lh, 28);

    Ok(ZipMember {
        data_offset: local_offset + 30 + name_len + extra_len,
        compressed_size,
        raw_size,
        method,
    })
}

/// Parse the npy header off the decompressed stream; returns the row count.
fn read_npy_header(r: &mut impl Read, raw_size: u64) -> io::Result<u64> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(bad("member is not an npy array"));
    }
    let (major, _minor) = (magic[6], magic[7]);
    let header_len = match major {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b) as u64
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            u32::from_le_bytes(b) as u64
        }
        v => return Err(bad(format!("unsupported npy version {v}"))),
    };
    let mut header = vec![0u8; header_len as usize];
    r.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header);

    if !header.contains(EXPECTED_DESCR) {
        return Err(bad(format!(
            "npy dtype mismatch.\n  expected: {EXPECTED_DESCR}\n  header:   {header}"
        )));
    }
    if header.contains("'fortran_order': True") {
        return Err(bad("fortran-ordered npy not supported"));
    }

    // shape is 1-D: "'shape': (26663697,)"
    let shape_pos = header
        .find("'shape': (")
        .ok_or_else(|| bad("npy header missing shape"))?;
    let rest = &header[shape_pos + 10..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let rows: u64 = digits
        .parse()
        .map_err(|_| bad("npy shape is not a 1-D integer"))?;

    // Cross-check: header + rows*64 must equal the member's raw size.
    let preamble = 8 + if major == 1 { 2 } else { 4 } + header_len;
    if preamble + rows * EVENT_SIZE as u64 != raw_size {
        return Err(bad(format!(
            "size mismatch: header {preamble} + {rows} rows x 64 != raw {raw_size}"
        )));
    }
    Ok(rows)
}

/// Streaming iterator over the events in an hftbacktest `.npz` file.
pub struct NpzEventReader {
    stream: BufReader<Box<dyn Read>>,
    rows: u64,
    rows_read: u64,
}

impl NpzEventReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let member = locate_member(&mut f)?;

        f.seek(SeekFrom::Start(member.data_offset))?;
        let compressed = BufReader::with_capacity(1 << 20, f).take(member.compressed_size);
        let raw: Box<dyn Read> = match member.method {
            8 => Box::new(DeflateDecoder::new(compressed)),
            _ => Box::new(compressed),
        };
        let mut stream = BufReader::with_capacity(1 << 20, raw);

        let rows = read_npy_header(&mut stream, member.raw_size)?;
        Ok(Self { stream, rows, rows_read: 0 })
    }

    /// Total number of events in the file (from the npy header).
    pub fn rows(&self) -> u64 {
        self.rows
    }
}

impl Iterator for NpzEventReader {
    type Item = io::Result<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rows_read == self.rows {
            return None;
        }
        let mut buf = [0u8; EVENT_SIZE];
        match self.stream.read_exact(&mut buf) {
            Ok(()) => {
                self.rows_read += 1;
                let f64le = |o: usize| f64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
                let u64le = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
                let i64le = |o: usize| i64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
                Some(Ok(Event {
                    ev: u64le(0),
                    exch_ts: i64le(8),
                    local_ts: i64le(16),
                    px: f64le(24),
                    qty: f64le(32),
                    order_id: u64le(40),
                    ival: i64le(48),
                    fval: f64le(56),
                }))
            }
            Err(e) => Some(Err(bad(format!(
                "truncated stream at row {}/{}: {e}",
                self.rows_read, self.rows
            )))),
        }
    }
}
