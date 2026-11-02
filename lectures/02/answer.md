# 第2回 解答

## 演習

- 使ったモデル・設定（`result.json` の `config` から）：
{
  "base_model": "Qwen/Qwen3.5-0.8B",
  "revision": "2fc06364715b967f1860aea9cf38778875588b17",
  "seed": 42,
  "num_train_epochs": 20,
  "warmup_steps": 6,
  "lr": 0.0002,
  "lora_r": 16,
  "lora_alpha": 32,
  "max_seq_len": 512,
  "max_new_tokens": 96,
  "gen": {
    "do_sample": false
  },
  "work_dir": "/content/qlora_lab",
  "drive_dir": "/content/drive/MyDrive/aie2026/qlora_lab"
}
- 自分で学習を実行したか（`executed_training_myself`）：はい
- 学習前：`format_rate` =　0.09523809523809523 / `content_rate` =　0.0
- 学習後：`format_rate` =　1.0 / `content_rate` =　0.9047619047619048
- 退行した質問：なし
- 学習後に残った失敗と、その型（根拠なしの質問に断定した / 根拠ありの質問に確認要求で逃げた / 形式の崩れ）：
    失敗した型: 根拠ありの質問に確認要求で逃げた（あるいはFAQの別項目で誤判定した）、もしくは根拠なしの質問に断定した
    「質問を受け付けてもらえるのは何曜日の何時からですか」に対して、内容評価で一部の不一致が発生しているか、needs_review: True にしてしまったなどの不一致が1〜2 件残っていた。形式の崩れ（format_rate）は 100% 達成できていた。
- 知識を問う質問（セクション 6）の結果（学習前の分類 → 学習後の分類）と気づいたこと：LoRA学習によって「JSON形式で出力する」フォーマットは完全に身に付いたが、FAQにない知識を問われた際、必ずしもすべてに「needs_review: true」と答えられるわけではなく、FAQに似た別のキーワードに引っ張られてハルシネーションを起こしてしまうことがあった。
- 学習で変わったこと・変わらなかったこと：
    変わったこと: format_rateが著明に向上した。指示された JSONのキー構造やデータ型をほぼ完全に再現できるようになった。
	変わらなかったこと: 類似表現に騙されて無関係のFAQから情報を引っ張って断定してしまうハルシネーションが残った。

## 宿題

- 取り組んだレベル：発展
- 変えた設定やデータ：
    データ: glossary_sft.jsonl（116件の講座用語集データ）を使用。セクション4のSFTフォーマットに厳密に準拠させるため、{"answer": ..., "needs_review": false} の形式にした上でLoRA追加学習を実施。
	設定: (
    output_dir="/content/qlora_glossary_ckpt",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=6,
    logging_steps=10,
    save_steps=20,
    save_total_limit=1,
    max_length=512,
    packing=False,
    report_to="none",
    seed=42
)
- 結果（比較表や数字）：
    - 学習前：`format_rate` =　0.16666666666666666 / `content_rate` =　0.0
    - 学習後：`format_rate` =　1.0 / `content_rate` =　0.833333
- ファインチューニングが向く場面・向かない場面についての考察：
    モデルに対して、指定されたJSONスキーマで返答するといった特定のルールを覚え込ませるにはSFTが向いていると考える。今回の演習でも、小さなモデルで堅牢に出力構造を維持できるようになった。
    一方で、用語集にない未収録の単語を尋ねられた際、学習後のモデルはハルシネーションを起こしていた。このように、日々変化する知識や、厳密な真偽判断が必要な場合、SFTを直接用いるのは困難と考える。それらはRAGで推論時に根拠ドキュメントとしてプロンプトに与えるのが適している。