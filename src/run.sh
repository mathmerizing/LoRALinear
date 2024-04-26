# Description: Run the models (MLP and MLP+AE and MLP+LoRA) with different ranks.

rm -rf ../output
mkdir -p ../output

echo "Running baseline MLP model..."
python3 main.py --nn mlp >> ../output/mlp.txt
echo "Done.\n"

echo "Running baseline MLP+AE models..."
for rank in 1 2 4 8 16 32 64 128
do
    echo "  MLP+AE model with rank=$rank:"
    python3 main.py --nn mlp+ae --rank $rank >> ../output/mlp+ae_rank=$rank.txt
done
echo "Done.\n"

echo "Running MLP+LoRA models..."
for rank in 1 2 4 8 16 32 64 128
do
    echo "  MLP+LoRA model with rank=$rank:"
    python3 main.py --nn mlp+lora --rank $rank >> ../output/mlp+lora_rank=$rank.txt
done

