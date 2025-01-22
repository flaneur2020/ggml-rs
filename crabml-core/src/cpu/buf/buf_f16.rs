use std::borrow::Cow;
use std::slice;

use half::f16;

pub fn f16_buf_from_bytes<'a>(buf: &[u8]) -> Cow<'a, [f16]> {
    let len = buf.len();
    assert_eq!(
        len % std::mem::size_of::<f32>(),
        0,
        "Length of slice must be multiple of f32 size"
    );
    let new_len = len / std::mem::size_of::<f16>();
    let ptr = buf.as_ptr() as *const f16;
    let f16_buf = unsafe { slice::from_raw_parts(ptr, new_len) };
    f16_buf.into()
}

// it's slow to initialize a vec![f16::ZERO; buf_size], nearly 80~200ms on preparing kv cache.
// we can initialize a vec![0 as u16; buf_size] and reinterpret it into Vec<f16> to make it
// faster, please note that the zerod f16 is not f16::ZERO, but f16(0x0000), do not read the
// uninitialized data in this buf.
#[expect(clippy::uninit_vec)]
pub fn alloc_f16_buf(len: usize) -> Vec<f16> {
    let mut buf = Vec::with_capacity(len);
    unsafe { buf.set_len(len) };
    buf
}

pub fn dequantize_f16_buf(buf: &[f16], start: usize) -> impl Iterator<Item = f32> + '_ {
    buf.iter().skip(start).map(|x| x.to_f32())
}

pub fn quantize_f32_f16<'a>(buf: &[f32]) -> Cow<'a, [f16]> {
    buf.iter()
        .map(|x| f16::from_f32(*x))
        .collect::<Vec<_>>()
        .into()
}

pub fn vec_dot_f16_f16(a: &[f16], a_offset: usize, b: &[f16], b_offset: usize, len: usize) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
    {
        vec_dot_f16_f16_neon(a, a_offset, b, b_offset, len)
    }

    #[cfg(not(any(target_arch = "aarch64", target_feature = "fp16")))]
    {
        vec_dot_f16_f16_fallback(a, a_offset, b, b_offset, len)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_f16_f16_neon(
    a: &[f16],
    a_offset: usize,
    b: &[f16],
    b_offset: usize,
    k: usize,
) -> f32 {
    use crate::cpu::archutil::aarch64 as myaarch64;
    unsafe {
        let mut sumv0 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let mut sumv1 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let k_rounded = k - k % 16;
        for ki in (0..k_rounded).step_by(16) {
            let av0 = myaarch64::vld1q_f16(a.as_ptr().add(a_offset + ki));
            let bv0 = myaarch64::vld1q_f16(b.as_ptr().add(b_offset + ki));
            let av1 = myaarch64::vld1q_f16(a.as_ptr().add(a_offset + ki + 8));
            let bv1 = myaarch64::vld1q_f16(b.as_ptr().add(b_offset + ki + 8));
            sumv0 = myaarch64::vfmaq_f16(sumv0, av0, bv0);
            sumv1 = myaarch64::vfmaq_f16(sumv1, av1, bv1);
        }

        let mut sum = myaarch64::vaddvq_f16(sumv0) + myaarch64::vaddvq_f16(sumv1);
        for ki in k_rounded..k {
            sum += (a.get_unchecked(a_offset + ki) * b.get_unchecked(b_offset + ki)).to_f32();
        }
        sum
    }
}

pub fn vec_dot_f16_f16_fallback(
    a: &[f16],
    a_offset: usize,
    b: &[f16],
    b_offset: usize,
    len: usize,
) -> f32 {
    let ac = &a[a_offset..a_offset + len];
    let bc = &b[b_offset..b_offset + len];
    let mut sum = 0.0;
    for i in 0..len {
        sum += ac[i].to_f32() * bc[i].to_f32();
    }
    sum
}

pub fn vec_dot_f16_f16_strided(
    a: &[f16],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f16],
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        vec_dot_f16_f16_strided_neon(a, a_base, a_stride, k, b)
    }

    #[cfg(not(any(target_arch = "aarch64",)))]
    {
        vec_dot_f16_f16_strided_fallback(a, a_base, a_stride, k, b)
    }
}

pub fn vec_fma_f16_f16(v: &[f16], b: f16, c: &mut [f16], v_offset: usize, m: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        vec_fma_f16_f16_neon(v, b, c, v_offset, m)
    }

    #[cfg(not(any(target_arch = "aarch64",)))]
    {
        vec_fma_f16_f16_fallback(v, b, c, v_offset, m)
    }
}

#[cfg(target_arch = "aarch64")]
fn vec_fma_f16_f16_neon(a: &[f16], b: f16, c: &mut [f16], a_offset: usize, m: usize) {
    use crate::cpu::archutil::aarch64 as myaarch64;
    unsafe {
        let m_rounded = m - m % 16;
        let bv = myaarch64::vdupq_n_f16(b.to_bits());
        for mi in (0..m_rounded).step_by(16) {
            let av0 = myaarch64::vld1q_f16(a.as_ptr().add(a_offset + mi));
            let av1 = myaarch64::vld1q_f16(a.as_ptr().add(a_offset + mi + 8));
            let cv0 = myaarch64::vld1q_f16(c.as_ptr().add(mi));
            let cv1 = myaarch64::vld1q_f16(c.as_ptr().add(mi + 8));
            let cv0 = myaarch64::vfmaq_f16(cv0, av0, bv);
            let cv1 = myaarch64::vfmaq_f16(cv1, av1, bv);
            myaarch64::vst1q_f16(c.as_mut_ptr().add(mi), cv0);
            myaarch64::vst1q_f16(c.as_mut_ptr().add(mi + 8), cv1);
        }
        for mi in m_rounded..m {
            c[mi] += a[a_offset + mi] * b;
        }
    }
}

#[allow(dead_code)]
fn vec_fma_f16_f16_fallback(a: &[f16], b: f16, c: &mut [f16], a_offset: usize, m: usize) {
    let m_rounded = m - m % 4;
    for mi in (0..m_rounded).step_by(4) {
        c[mi] += a[a_offset + mi] * b;
        c[mi + 1] += a[a_offset + mi + 1] * b;
        c[mi + 2] += a[a_offset + mi + 2] * b;
        c[mi + 3] += a[a_offset + mi + 3] * b;
    }
    for mi in m_rounded..m {
        c[mi] += a[a_offset + mi] * b;
    }
}

pub fn vec_convert_f16_f32(dst: &mut [f16], src: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    vec_convert_f16_f32_neon(dst, src);

    #[cfg(not(target_arch = "aarch64"))]
    dst.iter_mut().zip(src.iter()).for_each(|(d, s)| {
        *d = f16::from_f32(*s);
    });
}

#[cfg(target_arch = "aarch64")]
pub fn vec_convert_f16_f32_neon(dst: &mut [f16], src: &[f32]) {
    use std::arch::aarch64;

    use crate::cpu::archutil::aarch64 as myaarch64;

    dst.chunks_exact_mut(4)
        .zip(src.chunks_exact(4))
        .for_each(|(chunk_dst, chunk_src)| unsafe {
            let dst_ptr = chunk_dst.as_mut_ptr();
            let src_ptr = chunk_src.as_ptr();
            let src = std::arch::aarch64::vld1q_f32(src_ptr);
            let dst = myaarch64::vcvt_f32_f16(src);
            aarch64::vst1_u16(dst_ptr as *mut u16, dst as aarch64::uint16x4_t);
        })
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_f16_f16_strided_neon(
    a: &[f16],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f16],
) -> f32 {
    use crate::cpu::archutil::aarch64 as myaarch64;
    unsafe {
        let a_ptr = a.as_ptr().add(a_base);

        let mut sumv0 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let mut sumv1 = myaarch64::vdupq_n_f16(f16::ZERO.to_bits());
        let k_rounded = k - k % 16;
        for ki in (0..k_rounded).step_by(16) {
            let av_tmp = [
                *a_ptr.add(ki * a_stride),
                *a_ptr.add((ki + 1) * a_stride),
                *a_ptr.add((ki + 2) * a_stride),
                *a_ptr.add((ki + 3) * a_stride),
                *a_ptr.add((ki + 4) * a_stride),
                *a_ptr.add((ki + 5) * a_stride),
                *a_ptr.add((ki + 6) * a_stride),
                *a_ptr.add((ki + 7) * a_stride),
                *a_ptr.add((ki + 8) * a_stride),
                *a_ptr.add((ki + 9) * a_stride),
                *a_ptr.add((ki + 10) * a_stride),
                *a_ptr.add((ki + 11) * a_stride),
                *a_ptr.add((ki + 12) * a_stride),
                *a_ptr.add((ki + 13) * a_stride),
                *a_ptr.add((ki + 14) * a_stride),
                *a_ptr.add((ki + 15) * a_stride),
            ];
            let av0 = myaarch64::vld1q_f16(av_tmp.as_ptr());
            let bv0 = myaarch64::vld1q_f16(b.as_ptr().add(ki));
            let av1 = myaarch64::vld1q_f16(av_tmp.as_ptr().add(8));
            let bv1 = myaarch64::vld1q_f16(b.as_ptr().add(ki + 8));
            sumv0 = myaarch64::vfmaq_f16(sumv0, av0, bv0);
            sumv1 = myaarch64::vfmaq_f16(sumv1, av1, bv1);
        }

        let mut sum = myaarch64::vaddvq_f16(sumv0) + myaarch64::vaddvq_f16(sumv1);
        for ki in k_rounded..k {
            sum += (a.get_unchecked(a_base + ki * a_stride) * b.get_unchecked(ki)).to_f32();
        }
        sum
    }
}

pub fn vec_dot_f16_f16_strided_fallback(
    a: &[f16],
    a_base: usize,
    a_stride: usize,
    k: usize,
    b: &[f16],
) -> f32 {
    let mut sum: f16 = f16::ZERO;
    let k_rounded = k - k % 4;
    for ki in (0..k_rounded).step_by(4) {
        sum += a[a_base + ki * a_stride] * b[ki];
        sum += a[a_base + (ki + 1) * a_stride] * b[ki + 1];
        sum += a[a_base + (ki + 2) * a_stride] * b[ki + 2];
        sum += a[a_base + (ki + 3) * a_stride] * b[ki + 3];
    }
    for ki in (k_rounded..k).step_by(1) {
        sum += a[a_base + ki * a_stride] * b[ki];
    }
    sum.to_f32()
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::cpu::buf::buf_f16::vec_fma_f16_f16;

    #[test]
    fn test_vec_fma_f16_f16() {
        let a = vec![f16::from_f32(1.0); 16];
        let mut c = vec![f16::from_f32(0.0); 16];
        let b = f16::from_f32(2.0);
        vec_fma_f16_f16(&a, b, &mut c, 0, 16);
        assert_eq!(c, vec![f16::from_f32(2.0); 16]);

        vec_fma_f16_f16(&a, b, &mut c, 0, 16);
        assert_eq!(c, vec![f16::from_f32(4.0); 16]);
    }
}
