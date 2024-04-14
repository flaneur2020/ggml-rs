use std::cell::RefCell;
use std::rc::Rc;

use super::tokenizer_gpt2::Gpt2Tokenizer;
use super::tokenizer_llama::LlamaTokenizer;
use crate::error::Result;

pub type TokenID = usize;

pub struct Tokenizer {
    tokens: Rc<Vec<String>>,
    eos_token: TokenID,
    inner: TokenizerInner,
    decode_buf: RefCell<Utf8Buf>,
}

pub enum TokenizerInner {
    Llama(LlamaTokenizer),
    GPT2(Gpt2Tokenizer),
}

impl Tokenizer {
    pub fn new(
        tokens: Vec<String>,
        token_scores: Vec<f32>,
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let tokens = Rc::new(tokens);
        let tokenizer = LlamaTokenizer::new(tokens.clone(), token_scores, bos_token, eos_token);
        let decode_buf = RefCell::new(Utf8Buf::new());

        Self {
            tokens,
            eos_token,
            decode_buf,
            inner: TokenizerInner::Llama(tokenizer),
        }
    }

    pub fn vocab(&self) -> &[String] {
        &self.tokens
    }

    pub fn eos_token(&self) -> TokenID {
        self.eos_token
    }

    pub fn token(&self, token_id: TokenID) -> String {
        self.tokens[token_id].clone()
    }

    pub fn decode(&self, token: TokenID) -> Result<String> {
        let bytes = match &self.inner {
            TokenizerInner::Llama(inner) => inner.decode(token),
            TokenizerInner::GPT2(inner) => inner.decode(token),
        };
        Ok(self.decode_buf.borrow_mut().step(&bytes))
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<TokenID>> {
        match &self.inner {
            TokenizerInner::Llama(inner) => Ok(inner.encode(text, bos, eos, true)),
            TokenizerInner::GPT2(inner) => Ok(inner.encode(text, bos, eos, true)),
        }
    }
}

enum TokenizerKind {
    Llama,
    GPT2,
}

fn is_utf8_start(byte: u8) -> bool {
    // Check if the byte is not a continuation byte (10xxxxxx)
    (byte & 0b11000000) != 0b10000000
}

/// on the cases that a utf-8 character is split into multiple tokens, we need to buffer the tokens
/// until we have a valid utf-8 string, then return it.
pub struct Utf8Buf {
    buf: Vec<u8>,
}

impl Utf8Buf {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
        }
    }

    pub fn is_valid(&self) -> bool {
        std::str::from_utf8(&self.buf).is_ok()
    }

    pub fn step(&mut self, bytes: &[u8]) -> String {
        let utf8 = std::str::from_utf8(bytes);
        if utf8.is_ok() {
            return utf8.unwrap().to_string();
        }

        self.buf.extend(bytes);
        if self.is_valid() || self.buf.len() >= 4 {
            let s = String::from_utf8_lossy(&self.buf).to_string();
            self.buf.clear();
            return s;
        }

        "".to_string()
    }
}