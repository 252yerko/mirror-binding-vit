#!/bin/bash
# Опыт E (osf.io/ducms): зарегистрированные меры e_measure_full.py (md5 a5bd4b27…) на всех 12 обучениях условий 1–3.
# Контрольные точки (md5 сверены с логами Kaggle 20.11): зёрна 0–1 — ~/binding-vit-data/kaggle_s1/e_full (сессия 1, v2),
# зёрна 2–3 — ~/binding-vit-data/kaggle_s2/e_full (сессия 2, v3). Запуск из папки binding-vit:
#   caffeinate -i bash run_measure_ducms.sh
cd "$(dirname "$0")"
echo "md5 e_measure_full.py: $(md5 -q e_measure_full.py)  (ожидается a5bd4b271cbf96936846484a54667a6b)"
[ "$(md5 -q e_measure_full.py)" = "a5bd4b271cbf96936846484a54667a6b" ] || { echo "СТОП: e_measure_full.py изменён"; exit 1; }
for c in 1 2 3; do
  for s in 0 1 2 3; do
    if [ $s -le 1 ]; then CK=~/binding-vit-data/kaggle_s1/e_full; else CK=~/binding-vit-data/kaggle_s2/e_full; fi
    if [ -f "e_measure_full_c${c}_s${s}.csv" ]; then echo "== c${c}_s${s}: уже есть, пропуск"; continue; fi
    echo "== c${c}_s${s}  старт $(date '+%H:%M:%S')"
    python -u e_measure_full.py --cond $c --seed $s --ckpts "$CK" 2>&1 | tee "e_measure_full_c${c}_s${s}_log.txt"
    echo "== c${c}_s${s}  конец $(date '+%H:%M:%S'), код ${PIPESTATUS[0]}"
  done
done
echo "ВСЁ ГОТОВО $(date '+%H:%M:%S')"
