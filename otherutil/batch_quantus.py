metric = FaithfulnessCorrelation(
    model=model,
    explain_func=explain_func,  # 你的 Grad-CAM 或其他解释方法
    task="classification",
    device="cuda",
    return_instance_score=True
)

# 累积所有得分
all_scores = []

with torch.no_grad():
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # 使用 Quantus 指标
        scores = metric(inputs=x_batch, targets=y_batch)
        all_scores.extend(scores)  # 每个样本的得分

print("Average Score:", sum(all_scores) / len(all_scores))