#!/usr/bin/env python3
"""
Fetch and display the first SpaceX launch details.
"""
import requests


def main():
    """
    Fetch and display the first SpaceX launch details.
    """
    response = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    launches = response.json()

    first_launch = min(launches, key=lambda x: x['date_unix'])

    launch_name = first_launch['name']
    date_local = first_launch['date_local']
    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    rocket_response = requests.get(
        f'https://api.spacexdata.com/v4/rockets/{rocket_id}')
    rocket_name = rocket_response.json()['name']

    launchpad_response = requests.get(
        f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}')
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data['name']
    launchpad_locality = launchpad_data['locality']

    print(f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name}"
          f" ({launchpad_locality})")


if __name__ == "__main__":
    main()
