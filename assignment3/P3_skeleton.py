import pandas as pd
import time
import pickle

class streak:
    def __init__(self, l, r, v):
        self.l = l
        self.r = r
        self.v = v
        self.length = r - l + 1
    
    def dominates(self, compared_streak):
        return (self.length > compared_streak.length and self.v >= compared_streak.v) or \
            (self.length >= compared_streak.length and self.v > compared_streak.v)

def readfiles():    
    #read from the csv file and return a Pandas DataFrame.
    nba = pd.read_csv("1991-2004-nba.dat",  delimiter='#')
        
    #Pandas DataFrame allows you to select columns. 
    #We use column selection to split the data. 
    #We only need 2 columns in the data file: Player ID and Points.
    columns = ['ID', 'PTS']
    nba_records = nba[columns]
    
    #For each player, store the player's points in all games in an ordered list.
    #Store all players' sequences in a dictionary.
    pts = {}    
    cur_player = 'NULL'
    #The data file is already sorted by player IDs, followed by dates.
    for index, row in nba_records.iterrows():
        player, points = row
        if player != cur_player:
            cur_player = player
            pts[player] = []            
        pts[player].append(points)

    return pts


def prominent_streaks(sequences):
    #You algorithm goes here
    #You have the freedom to define any other functions that you deem necessary. 

    for sequence in sequences:
        lps = []
        potential_lps = []

        for k in range(len(sequence)):
            potential_lps.append(streak(0, k, sequence[k]))

            for candidate in candidate_streaks:
                if sequence[k] < candidate.v:
                    candidate.v = sequence[k]
                    candidate.r = k
                    candidate.length = candidate.r - candidate.l + 1
                elif s


    pass
    
    
# t0 = time.time()
# sequences = readfiles()
# t1 = time.time()
# print("Reading the data file takes ", t1-t0, " seconds.")

# with open("kobe", "wb") as f:
#     pickle.dump(sequences["BRYANKO01"], f)

with open("kobe", "rb") as f:
    sequences = pickle.load(f)

sequences = [sequences]

t1 = time.time()
streaks = prominent_streaks(sequences)
t2 = time.time()
print("Computing prominent streaks takes ", t2-t1, " seconds.")
print(streaks)
