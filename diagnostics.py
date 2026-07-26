import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figure_to_rgb(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(height, width, 3)
    plt.close(fig)
    return image


def semantic_map_image(map_inputs):
    # 诊断图保持与当前 2 通道地图输入一致，避免误读为旧的五通道版本。
    channel_titles = ["unknown", "frontier_heatmap"]
    fig, axes = plt.subplots(1, len(channel_titles), figsize=(5.2, 2.4), constrained_layout=True)
    for channel, axis, title in zip(map_inputs, axes, channel_titles):
        axis.imshow(channel, cmap="viridis", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    return figure_to_rgb(fig)


def node_feature_norm_overlay(belief, node_coords, feature_norm):
    fig, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    axis.imshow(belief, cmap="gray")
    scatter = axis.scatter(
        node_coords[:, 0],
        node_coords[:, 1],
        c=feature_norm,
        s=14,
        cmap="magma",
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("node map feature norm")
    axis.set_xlim(0, belief.shape[1])
    axis.set_ylim(belief.shape[0], 0)
    axis.axis("off")
    return figure_to_rgb(fig)


def action_probability_overlay(belief, node_coords, current_index, edge_indices, action_probs):
    fig, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    axis.imshow(belief, cmap="gray")
    current = node_coords[int(current_index)]
    if len(edge_indices) > 0:
        max_prob = max(float(np.max(action_probs)), 1e-8)
        for node_index, probability in zip(edge_indices, action_probs):
            target = node_coords[int(node_index)]
            strength = float(probability) / max_prob
            axis.plot(
                [current[0], target[0]],
                [current[1], target[1]],
                color=plt.cm.plasma(strength),
                linewidth=0.8 + 4.0 * strength,
                alpha=0.35 + 0.65 * strength,
            )
            axis.scatter(target[0], target[1], c=[plt.cm.plasma(strength)], s=24)
    axis.scatter(current[0], current[1], c="cyan", s=44, edgecolors="black", linewidths=0.5)
    axis.set_title("action probability")
    axis.set_xlim(0, belief.shape[1])
    axis.set_ylim(belief.shape[0], 0)
    axis.axis("off")
    return figure_to_rgb(fig)
