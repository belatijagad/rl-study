import numpy as np
import matplotlib.pyplot as plt

def visualize_policy(agent, usable_ace=True):
  fig = plt.figure(figsize=(15, 6))
  
  ax1 = fig.add_subplot(121, projection="3d")
  ax1.view_init(elev=30, azim=120)
  
  player_sum = np.arange(12, 22)
  dealer_showing = np.arange(1, 11)
  X, Y = np.meshgrid(player_sum, dealer_showing)
  
  Z = np.zeros_like(X, dtype=float)
  for i, dealer in enumerate(dealer_showing):
    for j, player in enumerate(player_sum):
      state = (player, dealer, usable_ace)
      Z[i, j] = np.max(agent.q_values[state])
  
  ax1.plot_surface(X, Y, Z, cmap="viridis")
  ax1.set_xlabel("Player sum")
  ax1.set_ylabel("Dealer showing")
  ax1.set_zlabel("Value")
  ax1.set_title(f"State values: {"With" if usable_ace else "Without"} usable ace")
  
  ax2 = fig.add_subplot(122)
  
  policy = np.zeros_like(X)
  for i, dealer in enumerate(dealer_showing):
    for j, player in enumerate(player_sum):
      state = (player, dealer, usable_ace)
      policy[i, j] = np.argmax(agent.q_values[state])
  
  im = ax2.imshow(policy, cmap="RdYlGn", aspect="auto", extent=[11.5, 21.5, 10.5, 0.5])
  ax2.set_xlabel("Player sum")
  ax2.set_ylabel("Dealer showing")
  ax2.set_title(f"Policy: {"With" if usable_ace else "Without"} usable ace")
  
  cbar = plt.colorbar(im)
  cbar.set_ticks([0, 1])
  cbar.set_ticklabels(["Stick", "Hit"])
  
  ax2.grid(True, color="black", linestyle="-", linewidth=0.5)
  
  plt.tight_layout()
  return fig