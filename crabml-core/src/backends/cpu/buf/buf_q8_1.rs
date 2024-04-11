use std::borrow::Cow;

use half::f16;

/// Q8_1 is only used as intermediate format for matmul on Q4_1, Q5_1 quantization. There's no need to implement
/// vec_dot for Q8_1. Compare to Q8_0, Q8_1 adds an extra `sum(d * qs[i])` value to the dot product
/// calculation. Take Q4_1 as example, it adds an extra `min` value than Q4_0. So calculating the dot product
/// of Q4_1 and Q8_1 is:
/// (a[0] - min) * b[0] * a.d * b.d + ... + (a[31] - min) * b[31] * a.d * b.d
/// = a[0] * b[0] * a.d * b.d + ... + a[31] * b[31] * a.d * b.d - min * (b[0] * a.d + ... + b[31] * a.d)
/// = (a[0] * b[0] + a[1] * b[1] + ... a[31] * b[31]) * a.d * b.d + min * b.s
/// = dot(a, b) * a.d * b.d + min * b.s
#[derive(Debug, Clone)]
pub struct QuantBufQ8_1<'a> {
    pub blocks: Cow<'a, [BlockQ8_1]>,
}

impl<'a> QuantBufQ8_1<'_> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let blk_size = std::mem::size_of::<BlockQ8_1>();
        assert_eq!(
            data.len() % blk_size,
            0,
            "data length must be a multiple of QuantBlockQ8_1 size"
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_1, data.len() / blk_size)
        };
        Self {
            blocks: blocks.into(),
        }
    }

    pub fn quantize(data: &[f32]) -> Self {
        let bs = quantize_f32_q8_1(data);
        Self { blocks: bs.into() }
    }

    fn blocks(&self) -> &[BlockQ8_1] {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len() * 32
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn dequantize(&'a self, start: usize) -> impl Iterator<Item = f32> + 'a {
        assert!(start % 32 == 0);

        let block_start = start / 32;
        self.blocks()[block_start..].iter().flat_map(|blk| {
            let mut buf = [0.0; 32];
            blk.dequantize(&mut buf);
            buf.into_iter()
        })
    }

    pub fn vec_dot(&self, a_offset: usize, b: &Self, b_offset: usize, len: usize) -> f32 {
        let abs = &self.blocks[a_offset / 32..(a_offset + len) / 32];
        let bbs = &b.blocks()[b_offset / 32..(b_offset + len) / 32];

        vec_dot_q8_1_q8_1(abs, bbs)
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct BlockQ8_1 {
    pub d: f16,       // delta
    pub s: f16,       // d * sum(qs[i])
    pub qs: [i8; 32], // quants
}

impl BlockQ8_1 {
    pub fn dequantize(&self, buf: &mut [f32]) {
        let d = f32::from(self.d);
        for (i, v) in buf.iter_mut().enumerate().take(32) {
            *v = self.qs[i] as f32 * d;
        }
    }
}

pub fn quantize_f32_q8_1(data: &[f32]) -> Vec<BlockQ8_1> {
    let mut bs = Vec::with_capacity(data.len() / 32);

    for chunk in data.chunks(32) {
        let mut max_abs_value = 0.0;

        // Find the maximum absolute value in the chunk
        for &value in chunk {
            let abs_value = value.abs();
            if abs_value > max_abs_value {
                max_abs_value = abs_value;
            }
        }

        let d = max_abs_value / 127.0; // Compute the scaling factor
        let mut qs = [0_i8; 32]; // Initialize the quantized values array
        let mut s = 0f32; // Initialize the sum of scaled values

        // Quantize the chunk
        for (i, &value) in chunk.iter().enumerate() {
            let scaled_value = value / d; // Scale the value
            // Convert the scaled value to i8, clamping it to the i8 range
            let quantized_value = scaled_value.max(i8::MIN as f32).min(i8::MAX as f32) as i8;
            qs[i] = quantized_value;
            s += quantized_value as f32; // Accumulate the sum of quantized values
        }

        s *= d; // Multiply the sum by d to get the final value of s

        // Store the block with the scaling factor, quantized values, and the sum
        bs.push(BlockQ8_1 {
            d: f16::from_f32(d),
            s: f16::from_f32(s),
            qs,
        });
    }

    bs
}

pub fn vec_dot_q8_1_q8_1(abs: &[BlockQ8_1], bbs: &[BlockQ8_1]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..bbs.len() {
        let mut sumi: i32 = 0;
        let ad = f32::from(abs[i].d);
        let bd = f32::from(bbs[i].d);
        for j in 0..32 {
            sumi += (abs[i].qs[j] as i32) * (bbs[i].qs[j] as i32);
        }
        sumf += sumi as f32 * ad * bd;
    }
    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_1_block() {
        assert_eq!(
            std::mem::size_of::<BlockQ8_1>(),
            2 * std::mem::size_of::<f32>() + 32,
            "wrong q8_1 block size/padding"
        );

        let mut buf: [u8; 80] = [0x1; 80];

        let d_bytes = f32::to_le_bytes(3.0);
        let s_bytes = f32::to_le_bytes(96.0);
        buf[0..4].copy_from_slice(&d_bytes);
        buf[4..8].copy_from_slice(&s_bytes);

        buf[8] = 2;
        buf[9] = 3;
        buf[10] = 4;
        buf[39] = 7;

        buf[40..44].copy_from_slice(&d_bytes);
        buf[44..48].copy_from_slice(&s_bytes);

        buf[48] = 2;
        buf[49] = 3;
        buf[50] = 4;
        buf[79] = 7;

        let blocks = QuantBufQ8_1::from_bytes(&buf[0..40]).blocks;
        assert_eq!(blocks[0].d.to_f32(), 3.0);
        assert_eq!(blocks[0].s.to_f32(), 96.0);
        assert_eq!(blocks[0].qs, [
            2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 7
        ]);

        let bf = QuantBufQ8_1::from_bytes(&buf);

        assert_eq!(bf.len(), 64);

        assert_eq!(bf.dequantize(0).collect::<Vec<_>>(), vec![
            6.0, 9.0, 12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0, 6.0, 9.0,
            12.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 21.0
        ]);
    }
}
