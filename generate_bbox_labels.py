import os.path as osp
import os
import pickle as pkl

if not osp.exists("bbox_labels"):
    os.mkdir("bbox_labels")
    

annotations_dir = "GroceryDataset_part2/BrandImagesFromShelves/"
save_dir = "bbox_labels"

classes = [str(x) for x in range(1,11)]

bbox_labels = {}

for cls in classes:
    
    cls_dir = osp.join(annotations_dir, cls)
    
    #Get the list of images
    class_ims = os.listdir(cls_dir)
    
    
    im_names = ["{}.JPG".format(name.split(".")[0]) for name in class_ims]
    
    im_names = [x for x in im_names if not(x[:6] == "C1_P01" or x[:6] == "C2_P01")]

    
    box_cords = [(name.split(".")[1].split("_")[-4:]) for name in class_ims]
    
    for i, name in enumerate(im_names):
        
        try:
            bbox_labels[name]
        except KeyError:
            bbox_labels[name] = []
            
        try:
            box_cord = [int(a) for a in box_cords[i]]
        except ValueError:
            continue
            
        
        box_cord[2], box_cord[3] = box_cord[0] + box_cord[2], box_cord[1] + box_cord[3]
        
        bbox_labels[name].append((int(cls), box_cord))
        
pkl.dump(bbox_labels, open("bboxes.pkl", "wb"))
    
        
        
            
    
    
    
    
