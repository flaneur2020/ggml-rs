use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::cell::RefCell;

use half::f16;
use rayon::prelude::*;

use crate::backends::cpu::buf::buf_f16::quantize_f32_f16;
use crate::backends::cpu::buf::buf_f16::vec_dot_f16_f16;
use crate::backends::cpu::buf::buf_f32::vec_dot_f32_f32_strided;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::TensorStrider;

thread_local! {
    pub static DESTRIDE_BUF: RefCell<Vec<f16>> = RefCell::new(vec![f16::ZERO; 64000]);
}

// (b, m, k) @ (b, k, ) -> (b, m, )
// a is allowed to be not contiguous, but not quantized
pub fn batch_matmul_vec<'a>(
    _device: &CpuTensorDeviceRef<'a>,
    a: &CpuTensorBuf<'a>,
    b: &CpuTensorBuf<'a>,
    c: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(strider1.shape().len() == 3);
    assert!(strider2.shape().len() == 2);
    assert!(strider1.shape()[0] == strider2.shape()[0]);
    assert!(strider1.shape()[2] == strider2.shape()[1]);
    assert!(strider2.is_contiguous());

    let bufc = c.as_f32_mut();

    let m = strider1.shape()[1];
    let k = strider1.shape()[2];
    let bi_stride = strider1.strides()[0];
    let mi_stride = strider1.strides()[1];
    let ki_stride = strider1.strides()[2];

    match a {
        CpuTensorBuf::F32(bufa) => {
            let bufb = b.as_f32_ref();
            batch_matmul_vec_f32(bufa, bufb, bufc, m, k, bi_stride, mi_stride, ki_stride);
        }
        CpuTensorBuf::F16(bufa) => {
            let bufb = b.as_f32_ref();
            let bufb = quantize_f32_f16(bufb);
            batch_matmul_vec_f16(bufa, &bufb, bufc, m, k, bi_stride, mi_stride, ki_stride);
        }

        _ => return Err((ErrorKind::TensorError, "a must be f32 or 16").into()),
    };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_vec_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    bi_stride: usize,
    mi_stride: usize,
    ki_stride: usize,
) {
    c.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
        let mi = i % m;
        let bi = (i - mi) / m;
        *bufcp = vec_dot_f32_f32_strided(
            a,
            bi * bi_stride + mi * mi_stride,
            ki_stride,
            k,
            &b[bi * k..(bi + 1) * k],
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn batch_matmul_vec_f16(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    m: usize,
    k: usize,
    a_stride0: usize,
    a_stride1: usize,
    a_stride2: usize,
) {
    if a_stride2 == 1 {
        c.par_iter_mut().enumerate().for_each(|(i, bufcp)| {
            let mi = i % m;
            let bi = (i - mi) / m;
            *bufcp = vec_dot_f16_f16(
                a,
                bi * a_stride0 + mi * a_stride1,
                &b[bi * k..(bi + 1) * k],
                0,
                k,
            );
        });
    } else {
        let chunk = 8;
        c.par_chunks_exact_mut(chunk)
            .enumerate()
            .for_each(|(i, bufcp)| {
                let mi = i * chunk % m;
                let bi = (i * chunk - mi) / m;
                let mut tmp = DESTRIDE_BUF.take();
                copy_strided_3d(
                    &a[bi * a_stride0 + mi * a_stride1..],
                    &[1, chunk, k],
                    &[a_stride0, a_stride1, a_stride2],
                    &mut tmp,
                );
                for j in 0..chunk {
                    bufcp[j] = vec_dot_f16_f16(&tmp, j * k, &b[bi * k..(bi + 1) * k], 0, k);
                }
                DESTRIDE_BUF.replace(tmp);
            })
    }
}

fn copy_strided_3d<T: Copy>(a: &[T], a_shape: &[usize], a_strides: &[usize], b: &mut [T]) {
    let mut pos = 0;
    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            for k in 0..a_shape[2] {
                b[pos] = a[i * a_strides[0] + j * a_strides[1] + k * a_strides[2]];
                pos += 1;
            }
        }
    }
}
