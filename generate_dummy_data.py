import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Sample data
games = [
    "Counter-Strike: Global Offensive", "Dota 2", "Team Fortress 2", 
    "Portal 2", "Half-Life 2", "Left 4 Dead 2", "Garry's Mod", 
    "The Elder Scrolls V: Skyrim", "Terraria", "Grand Theft Auto V",
    "Football Manager 2015", "Sid Meier's Civilization V", 
    "Borderlands 2", "Warframe", "Total War: ROME II - Emperor Edition"
]

user_ids = range(151603712, 151603712 + 100) # 100 users, starting with a common one from README

data = []
for uid in user_ids:
    # Each user plays 5-15 games
    n_games = np.random.randint(5, 16)
    played_games = np.random.choice(games, n_games, replace=False)
    for game in played_games:
        # Playtime between 1 and 500 hours
        playtime = np.random.uniform(1, 500)
        # behavior is 'play' (as per preprocess.py logic for steam-200k format)
        data.append([uid, game, "play", playtime, 0])
        # also add a 'purchase' record
        data.append([uid, game, "purchase", 1.0, 0])

# Randomly add some popular games to more users
popular_games = ["Counter-Strike: Global Offensive", "Dota 2", "Team Fortress 2"]
for game in popular_games:
    for uid in np.random.choice(user_ids, size=30, replace=False):
        if not any(d[0] == uid and d[1] == game for d in data):
             data.append([uid, game, "play", np.random.uniform(100, 1000), 0])
             data.append([uid, game, "purchase", 1.0, 0])

df = pd.DataFrame(data)
# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save as steam-200k.csv in the data folder
df.to_csv("data/steam-200k.csv", index=False, header=False)
print(f"Generated synthetic dataset with {len(df)} records at data/steam-200k.csv")
