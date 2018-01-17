import sys
import os
__stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import logging
import random
from collections import OrderedDict
import numpy as np
import hlt
sys.stdout = __stderr


HM_ENT_FEATURES = 5
PCT_CHANGE_CHANCE = 30
DESIRED_SHIP_COUNT = 20


game = hlt.Game("Training Dummy")
logging.info("Starting Training Dummy")

ship_plans = {}


def key_by_value(dictionary, value):
    for k, v in dictionary.items():
        if v[0] == value:
            return k
    return -99


def fix_data(data):
    new_list = []
    last_known_idx = 0
    for k in range(HM_ENT_FEATURES):
        # noinspection PyBroadException
        try:
            if k < len(data):
                last_known_idx = k
            new_list.append(data[last_known_idx])
        except:
            new_list.append(0)

    return new_list


LOSS = 0
while True:
    command_queue = []
    game_map = game.update_map()

    team_ships = game_map.get_me().all_ships()
    # noinspection PyProtectedMember
    all_ships = game_map._all_ships()
    # noinspection PyProtectedMember
    enemy_ships = [ship for ship in game_map._all_ships() if ship not in team_ships]

    my_ship_count = len(team_ships)
    enemy_ship_count = len(enemy_ships)
    all_ship_count = len(all_ships)

    my_id = game_map.get_me().id

    empty_planet_sizes = {}
    our_planet_sizes = {}
    enemy_planet_sizes = {}

    for p in game_map.all_planets():
        radius = p.radius
        if not p.is_owned():
            empty_planet_sizes[radius] = p
        elif p.owner.id == game_map.get_me().id:
            our_planet_sizes[radius] = p
        elif p.owner.id != game_map.get_me().id:
            enemy_planet_sizes[radius] = p

    hm_our_planets = len(our_planet_sizes)
    hm_empty_planets = len(empty_planet_sizes)
    hm_enemy_planets = len(enemy_planet_sizes)

    empty_planet_keys = sorted([k for k in empty_planet_sizes])[::-1]
    our_planet_keys = sorted([k for k in our_planet_sizes])[::-1]
    enemy_planet_keys = sorted([k for k in enemy_planet_sizes])[::-1]

    for ship in game_map.get_me().all_ships():
        try:
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                # Skip this ship
                continue

            shipid = ship.id
            change = False
            if random.randint(1, 100) <= PCT_CHANGE_CHANCE:
                change = True

            entities_by_distance = game_map.nearby_entities_by_distance(ship)
            entities_by_distance = OrderedDict(sorted(entities_by_distance.items(), key=lambda t: t[0]))

            closest_empty_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                     isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not
                                     entities_by_distance[distance][0].is_owned()]
            closest_empty_planet_distances = [distance for distance in entities_by_distance if
                                              isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not
                                              entities_by_distance[distance][0].is_owned()]

            closest_my_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                  isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and
                                  entities_by_distance[distance][0].is_owned() and (
                                          entities_by_distance[distance][0].owner.id == game_map.get_me().id)
                                  and ship.can_dock(entities_by_distance[distance][0])]
            closest_my_planets_distances = [distance for distance in entities_by_distance if
                                            isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and
                                            entities_by_distance[distance][0].is_owned() and (
                                                    entities_by_distance[distance][0].owner.id == game_map.get_me().id)
                                            and ship.can_dock(entities_by_distance[distance][0])]

            closest_enemy_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                     isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and
                                     entities_by_distance[distance][0] not in closest_my_planets and
                                     entities_by_distance[distance][0] not in closest_empty_planets]
            closest_enemy_planets_distances = [distance for distance in entities_by_distance if
                                               isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and
                                               entities_by_distance[distance][0] not in closest_my_planets and
                                               entities_by_distance[distance][0] not in closest_empty_planets]

            closest_team_ships = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                  isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and
                                  entities_by_distance[distance][0] in team_ships]
            closest_team_ships_distances = [distance for distance in entities_by_distance if
                                            isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and
                                            entities_by_distance[distance][0] in team_ships]

            closest_enemy_ships = [entities_by_distance[distance][0] for distance in entities_by_distance if
                                   isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and
                                   entities_by_distance[distance][0] not in team_ships]
            closest_enemy_ships_distances = [distance for distance in entities_by_distance if
                                             isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and
                                             entities_by_distance[distance][0] not in team_ships]

            largest_empty_planet_distances = []
            largest_our_planet_distances = []
            largest_enemy_planet_distances = []

            for i in range(HM_ENT_FEATURES):
                # noinspection PyBroadException
                try:
                    largest_empty_planet_distances.append(
                        key_by_value(entities_by_distance, empty_planet_sizes[empty_planet_keys[i]]))
                except:
                    largest_empty_planet_distances.append(-99)
                # noinspection PyBroadException
                try:
                    largest_our_planet_distances.append(
                        key_by_value(entities_by_distance, our_planet_sizes[our_planet_keys[i]]))
                except:
                    largest_our_planet_distances.append(-99)
                # noinspection PyBroadException
                try:
                    largest_enemy_planet_distances.append(
                        key_by_value(entities_by_distance, enemy_planet_sizes[enemy_planet_keys[i]]))
                except:
                    largest_enemy_planet_distances.append(-99)

            entity_lists = [fix_data(closest_empty_planet_distances),
                            fix_data(closest_my_planets_distances),
                            fix_data(closest_enemy_planets_distances),
                            fix_data(closest_team_ships_distances),
                            fix_data(closest_enemy_ships_distances),
                            fix_data(empty_planet_keys),
                            fix_data(our_planet_keys),
                            fix_data(enemy_planet_keys),
                            fix_data(largest_empty_planet_distances),
                            fix_data(largest_our_planet_distances),
                            fix_data(largest_enemy_planet_distances)]

            input_vector = []

            for i in entity_lists:
                for ii in i[:HM_ENT_FEATURES]:
                    input_vector.append(ii)

            input_vector += [my_ship_count,
                             enemy_ship_count,
                             hm_our_planets,
                             hm_empty_planets,
                             hm_enemy_planets]
            if my_ship_count > DESIRED_SHIP_COUNT:
                # ATTACK ENEMY CUZ TOO MANY SHIPS
                output_vector = 3 * [0]  # [0,0,0]
                output_vector[0] = 1  # [1,0,0]
                ship_plans[ship.id] = [1, 0, 0]

            elif change or ship.id not in ship_plans:
                # PICK A NEW PLAN
                # output_vector = AI.predict(np.array(input_vector).reshape([-1, 60]))
                output_vector = 3 * [0]
                output_vector[random.randint(0, 2)] = 1
                ship_plans[ship.id] = output_vector

            else:
                # DO WHAT EVERY YOU HAVE BEEN DOING
                output_vector = ship_plans[ship.id]

            try:
                # ATTACK ENEMY SHIP
                if np.argmax(output_vector) == 0:  # [1,0,0]
                    if not closest_enemy_ships:
                        continue
                    if not isinstance(closest_enemy_ships[0], int):
                        navigate_command = ship.navigate(
                            ship.closest_point_to(closest_enemy_ships[0]),
                            game_map,
                            speed=int(hlt.constants.MAX_SPEED),
                            ignore_ships=False)

                        if navigate_command:
                            command_queue.append(navigate_command)
                    else:
                        LOSS += 10

                # MINE ONE OF OUR PLANETS
                elif np.argmax(output_vector) == 1:
                    if not closest_my_planets:
                        continue
                    if not isinstance(closest_my_planets[0], int):
                        target = closest_my_planets[0]
                        # noinspection PyProtectedMember
                        if len(target._docked_ship_ids) < target.num_docking_spots:
                            if ship.can_dock(target):
                                command_queue.append(ship.dock(target))
                            else:
                                navigate_command = ship.navigate(
                                    ship.closest_point_to(target),
                                    game_map,
                                    speed=int(hlt.constants.MAX_SPEED),
                                    ignore_ships=False)

                                if navigate_command:
                                    command_queue.append(navigate_command)
                    else:
                        LOSS += 10

                # FIND AND MINE AN EMPTY PLANET
                elif np.argmax(output_vector) == 2:
                    if not closest_empty_planets:
                        continue
                    if not isinstance(closest_empty_planets[0], int):
                        target = closest_empty_planets[0]
                        if ship.can_dock(target):
                            command_queue.append(ship.dock(target))
                        else:
                            navigate_command = ship.navigate(
                                ship.closest_point_to(target),
                                game_map,
                                speed=int(hlt.constants.MAX_SPEED),
                                ignore_ships=False)

                            if navigate_command:
                                command_queue.append(navigate_command)
                    else:
                        LOSS += 10

            except Exception as e:
                logging.info(str(e))

        except Exception as e:
            logging.info(str(e))

    game.send_command_queue(command_queue)
