import os
import random
import shutil
import time
import traceback
from datetime import datetime

ship_requirement = 10
damage_requirement = 1000


def get_ships(data):
    return int(data.split("producing ")[1].split(" ships")[0])


def get_damage(data):
    return int(data.split("dealing ")[1].split(" damage")[0])


def get_rank(data):
    return int(data.split("rank #")[1].split(" and")[0])


def find_line(data):
    for idx, line in enumerate(data):
        if 'and was last alive on frame' in line:
            return idx


def get_data(hm):
    # noinspection PyBroadException
    try:
        data = []
        f = open('data.gameout', 'r')
        contents = f.readlines()
        f.close()
        ln = find_line(contents)
        for k in range(hm):
            data.append(contents[ln + k][:-2])
        return data
    except:
        time.sleep(1)
        get_data(hm)


def cleanup():
    # noinspection PyBroadException
    try:
        if os.path.exists('model/TrainingBot_1'):
            shutil.rmtree('model/TrainingBot_1')
        if os.path.exists('model/TrainingBot_2'):
            shutil.rmtree('model/TrainingBot_2')
    except:
        time.sleep(1)
        cleanup()


AI_wins = 0
dummy_wins = 0
i = 0

map_sizes = ['312 208',
             '240 160',
             '336 224',
             '360 240',
             '288 192',
             '264 176',
             '288 192']

bots = ['python MyBot.py TrainingBot_1',
        'python MyBot.py TrainingBot_2',
        'python SentdeBot.py',
        'python Training_Dummy.py']

while True:
    try:
        if os.path.isfile('data.gameout'):
            os.system('del /f data.gameout')
        if len([True for i in os.listdir('./') if i.endswith('.log')]) > 0:
            os.system('del /f *.log')
        winner = None
        if AI_wins > 0 or dummy_wins > 0:
            p1_pct = round(AI_wins / (AI_wins + dummy_wins) * 100.0, 2)
            p2_pct = round(dummy_wins / (AI_wins + dummy_wins) * 100.0, 2)
            print("{}, Games {}, AI Wins: {} WR: {}, Dummy Wins: {} WR: {}".format(datetime.now(), i, AI_wins, p1_pct,
                                                                                   dummy_wins, p2_pct))
        cleanup()

        if i % 2 == 0:
            cmd = 'activate.bat && halite.exe -d "{}" "{}" "{}" --replaydirectory "replays" >> data.gameout'.format(
                random.choice(map_sizes), bots[0], bots[1])
            print('Starting 1v1 Game')
            os.system(cmd)
            player = get_data(2)
            print(player[0])
            print(player[1])

            AI1_ships = get_ships(player[0])
            AI1_dmg = get_damage(player[0])
            AI1_rank = get_rank(player[0])

            AI2_ships = get_ships(player[1])
            AI2_dmg = get_damage(player[1])
            AI2_rank = get_rank(player[1])

            if AI1_rank == 1:
                winner = 'TrainingBot_1'
            elif AI2_rank == 1:
                winner = 'TrainingBot_2'
        else:
            cmd = 'activate.bat && halite.exe -d "{}" "{}" "{}" "{}" "{}" --replaydirectory "replays" >> data.gameout' \
                .format(random.choice(map_sizes), bots[0], bots[1], bots[2], bots[3])
            print('Starting 4 People DeathMatch')
            os.system(cmd)
            player = get_data(4)
            print(player[0])
            print(player[1])
            print(player[2])
            print(player[3])

            AI1_ships = get_ships(player[0])
            AI1_dmg = get_damage(player[0])
            AI1_rank = get_rank(player[0])

            AI2_ships = get_ships(player[1])
            AI2_dmg = get_damage(player[1])
            AI2_rank = get_rank(player[1])

            Dummy1_ships = get_ships(player[2])
            Dummy1_dmg = get_damage(player[2])
            Dummy1_rank = get_rank(player[2])

            Dummy2_ships = get_ships(player[3])
            Dummy2_dmg = get_damage(player[3])
            Dummy2_rank = get_rank(player[3])

            if Dummy1_rank == 1 or Dummy2_rank == 1:
                dummy_wins += 1
            else:
                AI_wins += 1
                if AI1_rank == 1:
                    winner = 'TrainingBot_1'
                elif AI2_rank == 1:
                    winner = 'TrainingBot_2'

        time.sleep(1.5)
        if winner is not None:
            if winner == 'TrainingBot_1' and AI1_ships < ship_requirement and AI1_dmg < damage_requirement:
                i += 1
                print('Not Saving because Winner {} Do not have requirements'.format(winner))
                continue
            if winner == 'TrainingBot_2' and AI2_ships < ship_requirement and AI2_dmg < damage_requirement:
                i += 1
                print('Not Saving because Winner {} Do not have requirements'.format(winner))
                continue
            if os.path.exists('model/{}'.format(winner)):
                if os.path.exists('model/ProjectFaker'):
                    shutil.rmtree('model/ProjectFaker')
                shutil.move('model/{}'.format(winner), 'model/ProjectFaker')
                for file in os.listdir('model/ProjectFaker'):
                    newname = str(file).replace(winner, 'ProjectFaker')
                    os.rename('model/ProjectFaker/' + file, 'model/ProjectFaker/' + newname)
                print('Model was Updated to {} weights'.format(winner))
        else:
            print('A dummy won the game shame on you AI =_=')
        i += 1
        print(' ')

    except Exception as e:
        print(str(e))
        traceback.print_exc()
        time.sleep(5)
        print(' ')
