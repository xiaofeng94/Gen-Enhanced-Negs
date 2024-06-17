import os
import json


json_file_list =[
    "odinw/AerialMaritimeDrone/large/valid/annotations_without_background.json",
    "odinw/AerialMaritimeDrone/tiled/valid/annotations_without_background.json",
    "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/annotations_without_background.json",
    "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json",
    "odinw/BCCD/BCCD.v3-raw.coco/valid/annotations_without_background.json",
    "odinw/boggleBoards/416x416AutoOrient/export/val_annotations_without_background.json",
    "odinw/brackishUnderwater/960x540/valid/annotations_without_background.json",
    "odinw/ChessPieces/Chess Pieces.v23-raw.coco/valid/annotations_without_background.json",
    "odinw/CottontailRabbits/valid/annotations_without_background.json",
    "odinw/dice/mediumColor/export/val_annotations_without_background.json",
    "odinw/DroneControl/Drone Control.v3-raw.coco/valid/annotations_without_background.json",
    "odinw/EgoHands/generic/valid/annotations_without_background.json",
    "odinw/EgoHands/specific/valid/annotations_without_background.json",
    "odinw/HardHatWorkers/raw/valid/annotations_without_background.json",
    "odinw/MaskWearing/raw/valid/annotations_without_background.json",
    "odinw/MountainDewCommercial/valid/annotations_without_background.json",
    "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/annotations_without_background.json",
    "odinw/openPoetryVision/512x512/valid/annotations_without_background.json",
    "odinw/OxfordPets/by-breed/valid/annotations_without_background.json",
    "odinw/OxfordPets/by-species/valid/annotations_without_background.json",
    "odinw/Packages/Raw/valid/annotations_without_background.json",
    "odinw/PascalVOC/valid/annotations_without_background.json",
    "odinw/pistols/export/val_annotations_without_background.json",
    "odinw/PKLot/640/valid/annotations_without_background.json",
    # "odinw/plantdoc/100x100/valid/annotations_without_background.json",
    "odinw/plantdoc/416x416/valid/annotations_without_background.json",
    "odinw/pothole/valid/annotations_without_background.json",
    "odinw/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json",
    "odinw/selfdrivingCar/fixedLarge/export/val_annotations_without_background.json",
    "odinw/ShellfishOpenImages/raw/valid/annotations_without_background.json",
    "odinw/ThermalCheetah/valid/annotations_without_background.json",
    "odinw/thermalDogsAndPeople/valid/annotations_without_background.json",
    "odinw/UnoCards/raw/valid/annotations_without_background.json",
    "odinw/VehiclesOpenImages/416x416/valid/annotations_without_background.json",
    "odinw/websiteScreenshots/valid/annotations_without_background.json",
    "odinw/WildfireSmoke/valid/annotations_without_background.json",
]


if __name__ == "__main__":

    categories = []
    image_count = 0

    for json_file in json_file_list:
        data = json.load(open(os.path.join("DATASET", json_file), 'r'))

        catInfo = data['categories']
        imgInfo = data['images']

        catNames = [x['name'] for x in catInfo]
        categories += catNames
        image_count += len(imgInfo)

    # import ipdb
    # ipdb.set_trace()
    categories = set(categories)

    print("# unique categories: %d"%(len(categories)))
    print("# images: %d"%(image_count))



