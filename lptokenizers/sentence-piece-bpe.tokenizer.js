/**
 * Huggingface Tokenizers Experiments
 * https://github.com/loretoparisi/hf-tokenizers-experiments
 * @author Loreto Parisi (loretoparisi at gmail dot com)
 * @2022 Loreto Parisi (loretoparisi at gmail dot com)
*/

"use strict";

const { promisify } = require("util");
const { SentencePieceBPETokenizer, BertWordPieceTokenizer, ByteLevelBPETokenizer, BPETokenizer } = require("tokenizers");
const { Tokenizer, EncodeOptions } = require("tokenizers/bindings/tokenizer");
const { nfkcNormalizer } = require("tokenizers/bindings/normalizers");
const { metaspacePreTokenizer } = require("tokenizers/bindings/pre-tokenizers");
const { metaspaceDecoder } = require("tokenizers/bindings/decoders");
const { getTokenContent } = require("tokenizers");
const { templateProcessing } = require("tokenizers/bindings/post-processors");
const { Encoding } = require("tokenizers/implementations/encoding");
const { TruncationStrategy, TruncationDirection, PaddingDirection } = require("tokenizers/bindings/enums");
const models = require("tokenizers/bindings/models");
        
class LPSentencePieceBPETokenizer extends SentencePieceBPETokenizer {
    constructor(tokenizer, configuration) {
        super(tokenizer, configuration);
        this.defaultTrainOptions = {
            initialAlphabet: [],
            limitAlphabet: 1000,
            minFrequency: 2,
            showProgress: true,
            specialTokens: ["<unk>", "<s>", "</s>", "<sep>", "<mask>"],
            vocabSize: 30000,
        };
    }
    static async fromOptions(options) {
        const opts = Object.assign(Object.assign({}, this.defaultOptions), options);
        const unkToken = getTokenContent(opts.unkToken);
        let model;
        if (opts.vocabFile && opts.mergesFile) {
            const modelOptions = {
                dropout: opts.dropout,
                unkToken: unkToken,
            };
            const fromFile = promisify(models.BPE.fromFile);
            model = await fromFile(opts.vocabFile, opts.mergesFile, modelOptions);
        }
        else {
            model = models.BPE.empty();
        }

        const tokenizer = new Tokenizer(model);
        for (const token of [
            opts.clsToken,
            opts.sepToken,
            opts.unkToken,
            opts.padToken,
            opts.maskToken,
        ]) {
            if (tokenizer.tokenToId(token) !== undefined) {
                tokenizer.addSpecialTokens([token]);
            }
        }
        tokenizer.setNormalizer(nfkcNormalizer());
        const preTokenizer = metaspacePreTokenizer(opts.replacement, opts.addPrefixSpace);
        tokenizer.setPreTokenizer(preTokenizer);
        const decoder = metaspaceDecoder(opts.replacement, opts.addPrefixSpace);
        tokenizer.setDecoder(decoder);
        const instance =  new LPSentencePieceBPETokenizer(tokenizer, opts);
        instance.setPostProcessor(templateProcessing(
            "<s> $A </s>",
            "<s> $A </s> $B:1 </s>:1",
            [
                ["<s>", instance.tokenToId("<s>")],
                ["</s>", instance.tokenToId("</s>")],
            ],
        ));

        // padding and truncation
        instance.setPadding({ maxLength: LPSentencePieceBPETokenizer.defaultOptions.maxLength });
        instance.setTruncation(LPSentencePieceBPETokenizer.defaultOptions.maxLength, { strategy: TruncationStrategy.LongestFirst });

        return instance;
    }
    async encode(sequence, pair, options) {
        console.log(sequence, pair, options);
        const encode = promisify(this.tokenizer.encode.bind(this.tokenizer));
        const rawEncoding = await encode(sequence, pair !== null && pair !== void 0 ? pair : null, options !== null && options !== void 0 ? options : null);
        return new Encoding(rawEncoding);
    }
    async encodeBatch(sequences, options) {
        console.log('encodeBatch')
        const encodeBatch = promisify(this.tokenizer.encodeBatch.bind(this.tokenizer));
        const rawEncodings = await encodeBatch(sequences, options);
        return rawEncodings.map((e) => new Encoding(e));
    }
    async decode(ids, skipSpecialTokens = true) {
        const decode = promisify(this.tokenizer.decode.bind(this.tokenizer));
        const decoded = await decode(ids, skipSpecialTokens);
        return decoded
    };
}
LPSentencePieceBPETokenizer.defaultOptions = {
    addPrefixSpace: true,
    replacement: "▁",
    unkToken: "<unk>",
    clsToken: "<s>",
    maskToken: "<mask>",
    padToken: "<pad>",
    sepToken: "</s>",
    maxLength: 512
};

module.exports.LPSentencePieceBPETokenizer = LPSentencePieceBPETokenizer;