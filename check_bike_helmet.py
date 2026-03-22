import json
with open("data/bicycle_helmet/train/_annotations.coco.json") as f:
    coco = json.load(f)

cat_map = {c["name"]: c["id"] for c in coco["categories"]}
helmet_id = cat_map.get("Helmet")
bike_id   = cat_map.get("Bicycle")

by_image = {}
for ann in coco["annotations"]:
    by_image.setdefault(ann["image_id"], set()).add(ann["category_id"])

has_helmet  = sum(1 for cats in by_image.values() if helmet_id in cats)
has_bicycle = sum(1 for cats in by_image.values() if bike_id   in cats)
has_both    = sum(1 for cats in by_image.values() if helmet_id in cats and bike_id in cats)

print(f"Has Bicycle: {has_bicycle}/{len(by_image)}")
print(f"Has Helmet:  {has_helmet}/{len(by_image)}")
print(f"Has both:    {has_both}/{len(by_image)}")