//! Tool argument normalization shared by UI adapters.

use serde::Serialize;
use serde_json::{
    Value,
    ser::{CompactFormatter, Formatter},
};
use std::io::{self, Write};

#[must_use]
pub fn normalize_tool_input(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

/// Retain serialized strings; encode other arguments as Python json.dumps does.
#[must_use]
pub fn normalize_tool_args(input: &Value) -> String {
    match input {
        Value::String(raw) => raw.clone(),
        Value::Null => "{}".to_owned(),
        _ => python_json_dumps(input).expect("JSON values serialize to memory"),
    }
}

/// Encode Python's spaced, ASCII JSON form while retaining struct field order.
pub(crate) fn python_json_dumps<T: Serialize + ?Sized>(value: &T) -> Result<String, crate::Error> {
    let mut bytes = Vec::new();
    value.serialize(&mut serde_json::Serializer::with_formatter(
        &mut bytes,
        PythonFormatter,
    ))?;
    Ok(String::from_utf8(bytes).expect("JSON serializer emits UTF-8"))
}

/// Python's truth value for values carried over JSON.
pub(crate) fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

struct PythonFormatter;
impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        let mut raw = Vec::new();
        CompactFormatter.write_f64(&mut raw, value)?;
        write_python_float(
            writer,
            std::str::from_utf8(&raw).expect("JSON floats are ASCII"),
        )
    }

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        let mut raw = Vec::new();
        CompactFormatter.write_f32(&mut raw, value)?;
        write_python_float(
            writer,
            std::str::from_utf8(&raw).expect("JSON floats are ASCII"),
        )
    }

    fn write_string_fragment<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        let mut ascii_start = 0;
        for (offset, ch) in fragment.char_indices() {
            if ch < '\u{7f}' {
                continue;
            }
            writer.write_all(&fragment.as_bytes()[ascii_start..offset])?;
            for code in ch.encode_utf16(&mut [0; 2]).iter() {
                write!(writer, "\\u{code:04x}")?;
            }
            ascii_start = offset + ch.len_utf8();
        }
        writer.write_all(&fragment.as_bytes()[ascii_start..])
    }
}

/// Retain the serializer's shortest roundtripping digits and apply Python's layout.
/// Python uses fixed notation for exponents -4 through 15, and always writes a
/// decimal point for integral floats in that range. Scientific exponents carry
/// a sign and at least two digits.
fn write_python_float<W: ?Sized + Write>(writer: &mut W, raw: &str) -> io::Result<()> {
    let number = if let Some(number) = raw.strip_prefix('-') {
        writer.write_all(b"-")?;
        number
    } else {
        raw
    };
    let (mantissa, exponent) =
        number
            .split_once('e')
            .map_or((number, 0), |(mantissa, exponent)| {
                (
                    mantissa,
                    exponent
                        .parse::<i32>()
                        .expect("JSON serializer emits decimal exponents"),
                )
            });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits = mantissa.replace('.', "");
    let Some(leading) = digits.find(|digit| digit != '0') else {
        return writer.write_all(b"0.0");
    };
    let point = decimal + exponent - leading as i32;
    digits.drain(..leading);
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let exponent = point - 1;
    if !(-4..16).contains(&exponent) {
        writer.write_all(&digits.as_bytes()[..1])?;
        if digits.len() > 1 {
            writer.write_all(b".")?;
            writer.write_all(&digits.as_bytes()[1..])?;
        }
        return write!(writer, "e{exponent:+03}");
    }
    if point <= 0 {
        writer.write_all(b"0.")?;
        for _ in 0..-point {
            writer.write_all(b"0")?;
        }
        writer.write_all(digits.as_bytes())
    } else if point >= digits.len() as i32 {
        writer.write_all(digits.as_bytes())?;
        for _ in digits.len()..point as usize {
            writer.write_all(b"0")?;
        }
        writer.write_all(b".0")
    } else {
        writer.write_all(&digits.as_bytes()[..point as usize])?;
        writer.write_all(b".")?;
        writer.write_all(&digits.as_bytes()[point as usize..])
    }
}
