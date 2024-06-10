for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
    echo "======LAYER $i======="
    sed "s/index: $((i-1))/index: $i/g" example/bert768-upos-layer5-example-cpu.yaml
    sed "s/bert-base-inattn-$((i-1))-norm/bert-base-inattn-$i-norm/g" example/bert768-upos-layer5-example-cpu.yaml
    accelerate launch --multi_gpu --num_processes 3 vinfo/experiment.py example/bert768-upos-layer5-example-cpu.yaml
done
