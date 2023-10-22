use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;

use super::CpuTensor;
use super::arithmetic::add_inplace;
use super::buf::CpuTensorBuf;
use crate::error::ErrorKind;
use crate::error::Result;
use crate::tensor::tensor::TensorBackend;
use crate::tensor::tensor::TensorBufID;
use crate::tensor::tensor::TensorOp;
use crate::tensor::tensor::TensorOpVar;

pub struct CpuTensorPool<'a> {
    bufs: Rc<RefCell<HashMap<TensorBufID, CpuTensorBuf<'a>>>>,
    next_buf_id: usize,
}

pub struct PooledCpuTensorRef<'a> {
    buf_id: TensorBufID,
    tensor: Option<CpuTensor<'a>>,
    bufs: Rc<RefCell<HashMap<TensorBufID, CpuTensorBuf<'a>>>>,
}

impl<'a> Deref for PooledCpuTensorRef<'a> {
    type Target = CpuTensor<'a>;

    fn deref(&self) -> &Self::Target {
        self.tensor.as_ref().unwrap()
    }
}

impl<'a> DerefMut for PooledCpuTensorRef<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor.as_mut().unwrap()
    }
}

impl Drop for PooledCpuTensorRef<'_> {
    fn drop(&mut self) {
        let buf = self.tensor.take().unwrap().into_buf();
        self.bufs.borrow_mut().insert(self.buf_id, buf);
    }
}

impl<'a> CpuTensorPool<'a> {
    pub fn new() -> Self {
        Self {
            bufs: Rc::new(RefCell::new(HashMap::new())),
            next_buf_id: 0,
        }
    }

    pub fn load(&self, op_var: &TensorOpVar) -> Result<Option<CpuTensor>> {
        let bufs = self.bufs.borrow();
        let buf = bufs.get(&op_var.buf_id);
        let buf = match buf {
            Some(buf) => buf,
            None => return Ok(None),
        };

        let tensor = CpuTensor::new(buf.clone(), op_var.strider.clone())?;
        Ok(Some(tensor))
    }   
}

pub struct CpuTensorBackend<'a> {
    bufs: Vec<CpuTensorBuf<'a>>,
}

impl CpuTensorBackend<'_> {
    pub fn new() -> Self {
        Self { bufs: Vec::new() }
    }

    pub fn get_tensor(&self, op_var: &TensorOpVar) -> Result<CpuTensor> {
        let buf = self.bufs[op_var.buf_id].as_ref();
        CpuTensor::new(buf, op_var.strider.clone())
    }

    pub fn take(&self, op_var: &TensorOpVar) -> Result<CpuTensor> {
        let buf = self.bufs[op_var.buf_id].as_ref();
        CpuTensor::new(buf, op_var.strider.clone())
    }
}

impl<'a> TensorBackend<'a> for CpuTensorBackend<'a> {
    fn process_op(&mut self, op: TensorOp) -> Result<Option<TensorOpVar>> {
        match &op {
            TensorOp::AddInplace { lhs, rhs } => {
                let lhs = self.get_tensor(lhs)?;
                let rhs = self.get_tensor(rhs)?;
                add_inplace(lhs, &rhs);
            }
            _ => todo!(),
        }
        Ok(None)
    }

    fn import_buf(&mut self, buf: CpuTensorBuf<'a>) -> Result<TensorBufID> {
        let next_buf_id = self.bufs.len();
        self.bufs.push(buf);
        Ok(next_buf_id)
    }

    fn export_buf(self, buf_id: TensorBufID, dst: &mut [f32]) -> Result<()> {
        if buf_id >= self.bufs.len() {
            return Err((ErrorKind::InvalidArgs, "invalid buf_id").into());
        }
        let buf = &self.bufs[buf_id];
        let buf = match buf {
            CpuTensorBuf::F32(buf) => buf,
            _ => {
                return Err((
                    ErrorKind::InvalidArgs,
                    format!("only f32 buf can be exported, but got {}", buf.typ()),
                )
                    .into());
            }
        };

        if buf.len() != dst.len() {
            return Err((
                ErrorKind::InvalidArgs,
                format!(
                    "mismatched buf len, want {}, but got {}",
                    dst.len(),
                    buf.len()
                ),
            )
                .into());
        }

        dst.copy_from_slice(buf);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
    }
}