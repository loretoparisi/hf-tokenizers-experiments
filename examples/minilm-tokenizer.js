/**
 * Huggingface Tokenizers Experiments
 * https://github.com/loretoparisi/hf-tokenizers-experiments
 * @author Loreto Parisi (loretoparisi at gmail dot com)
 * @2022 Loreto Parisi (loretoparisi at gmail dot com)
*/

"use strict";

const { promisify } = require("util");

(async function () {
    
    const { LPSentencePieceBPETokenizer } = require("../lptokenizers/sentence-piece-bpe.tokenizer");
    
    var lpTokenizer = await LPSentencePieceBPETokenizer.fromOptions({
        padMaxLength: false,
        vocabFile: "../vocab/minilm/minilm-vocab.json"
        , mergesFile: "../vocab/minilm/minilm-merges.txt"
    });

    let encoder = (tokenizer) => promisify(tokenizer.encode.bind(tokenizer))
    let encoderBatch = (tokenizer) => promisify(tokenizer.encodeBatch.bind(tokenizer))
    let decoder = (tokenizer) => promisify(tokenizer.decode.bind(tokenizer))

    let skipSpecialTokens = true;
    let addSpecialTokens = true;
    let isPretokenized = false;

    var encode, encodeBatch, decode, encoded, decoded;

    encode = encoder(lpTokenizer);
    encodeBatch = encoderBatch(lpTokenizer);
    decode = decoder(lpTokenizer);

    encoded = await encode("Hello how are you?");
    console.log("ids ", encoded.getIds());
    console.log("tokens ", encoded.getTokens());
    decoded = await lpTokenizer.decode(encoded.getIds(), skipSpecialTokens);
    console.log("decoded - ", decoded); // hello how are you?
    decoded = await lpTokenizer.decode(encoded.getIds(), !skipSpecialTokens);
    console.log("decoded - ", decoded); // hello how are you?
    //[0, 35378, 3642, 621, 398, 32, 2]
    var output = await encodeBatch(
        [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
    );
    console.log(output[0].getTokens());
    console.log(output[1].getTokens());
    console.log(output[1].getAttentionMask());
    
    const sentence = "Hello how are you tonight?";
    encoded = await lpTokenizer.encode(sentence, null, {
        addSpecialTokens: addSpecialTokens,
        isPretokenized: isPretokenized
    });
    console.log("ids ", encoded.getIds());
    console.log("tokens ", encoded.getTokens());
    decoded = await lpTokenizer.decode(encoded.getIds(), skipSpecialTokens);
    console.log("decoded - ", decoded); // hello how are you?
    decoded = await lpTokenizer.decode(encoded.getIds(), !skipSpecialTokens);
    console.log("decoded (special tokens)- ", decoded); // hello how are you?

}).call(this);