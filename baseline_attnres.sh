for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
    echo "======LAYER $i======="
    python3 vinfo/experiment.py example/bert-attn-res-ln-configs/ptb_pos/baseline/bert768-upos-layer$i-example-gpu.yaml
done
