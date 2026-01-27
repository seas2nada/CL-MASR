OUTPUT_DIR=$1
OUTPUT_DIR=$PWD/$OUTPUT_DIR

python analyze_logs.py $OUTPUT_DIR

# python summarize_final_wer.py $OUTPUT_DIR \
#   --last_locale ia \
#   --base_locales "en,zh-CN,de,es,ru,fr,pt,ja,tr,pl" \
#   --new_locales "ab,ckb,eo,fy-NL,ia,kab,kmr,lg,mhr,rw"

python summarize_final_wer.py $OUTPUT_DIR \
  --last_locale ia \
  --base_locales "en,zh-CN,de,es,ru,fr,pt,ja,tr,pl" \
  --new_locales "ab,ckb,eo,fy-NL,ia,kab,kmr,lg,mhr,rw"