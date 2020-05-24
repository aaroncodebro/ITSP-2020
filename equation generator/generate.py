import numpy as np
from os import listdir
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize, rotate
from skimage.util import random_noise, invert
from skimage.color import gray2rgb
import math
import os
import json
import cv2

path_parent = os.path.dirname(os.getcwd())
read_path = os.path.join(path_parent,"dataset/extracted_images")
train = True
if train:
    write_single_path = os.path.join(path_parent,"/normalized/train")
    formula_path = os.path.join(path_parent,"formulas/train/fractions")
    os.makedirs(write_single_path)
    os.makedirs(formula_path)
else:
    write_single_path = os.path.join(path_parent,"/normalized/test")
    formula_path = os.path.join(path_parent,"formulas/test/fractions")
    os.makedirs(write_single_path)
    os.makedirs(formula_path)

label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', 'leq', 'neq', 'geq', 'alpha','beta', 'lambda', 'lt', 'gt', 'x', 'y']
label_names_dict = dict()
for label,no in zip(label_names,list(range(len(label_names)))):
    label_names_dict[label] = no
    if not os.path.exists(write_single_path+"/"+label):
        os.makedirs(write_single_path+"/"+label)
        
    
print(label_names_dict)

def crop(img):
    crop = np.copy(img)/255
    h,w = img.shape
    left = 0
    while left < w//2 and np.sum(crop[:,left]) >= 0.98*h:
        left += 1
    right = w-1
    while right > w//2 and np.sum(crop[:,right]) >= 0.98*h:
        right -= 1
    if left > 0:
        left -1
    if right < h-1:
        right += 1
    crop = crop[:,left:right]
    
    top = 0
    while top < h//2 and np.sum(crop[top,:]) >= 0.98*w:
        top += 1
    bottom = h-1
    while bottom > h//2 and np.sum(crop[bottom,:]) >= 0.98*w:
        bottom -= 1
    if top > 0:
        top -= 1
    if bottom < h-1:
        bottom += 1
    crop = crop[top:bottom,:]*255
    return crop

def add_symbol_to_image(img,folder,choices,padding,minsize,maxsize,bpower=False,bsmall=False,bnom=False,bden=False,
                        width=False):
    choice = np.random.randint(len(choices))
    symbol_img = io.imread(read_path+"/"+folder+"/"+choices[choice])
    new_width = np.random.randint(minsize,maxsize+1)
    new_height = np.random.randint(minsize,maxsize+1)
    if width is not False:
        new_width = width
    symbol_img_res = resize(symbol_img, (new_height, new_width), cval=1)*255
    symbol_img_res = crop(symbol_img_res)
    new_height, new_width = symbol_img_res.shape
    shift = np.random.randint(-4+(60-new_height)//2,4+(60-new_height)//2)

    
    bounding_box = {
        'xmin': padding,
        'xmax': padding+new_width,
        'ymin': 65+shift-15*bpower+10*bsmall-30*bnom+30*bden,
        'ymax': 65+shift+new_height-15*bpower+10*bsmall-30*bnom+30*bden,
        'class_text': folder,
        'class': label_names_dict[folder]
    }
    
    if folder == "y" or folder == "beta":
        bounding_box['ymin'] += 10
        bounding_box['ymax'] += 10
                       
    
    xmin, xmax = bounding_box['xmin'],bounding_box['xmax']
    ymin, ymax = bounding_box['ymin'],bounding_box['ymax']
    
    img[ymin:ymax,xmin:xmax] += invert(symbol_img_res)+254
    padding += new_width+np.random.randint(2,5)
    
    return img,padding,bounding_box

def add_rectangles(img, bounding_boxes):
    img_color = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    for bounding_box in bounding_boxes[1:]:
        xmin, xmax = bounding_box['xmin'], bounding_box['xmax']
        ymin, ymax = bounding_box['ymin'], bounding_box['ymax']
        img_color[ymin,xmin:xmax] = [255,0,0]
        img_color[ymax,xmin:xmax] = [255,0,0]
        img_color[ymin:ymax,xmin] = [255,0,0]
        img_color[ymin:ymax,xmax] = [255,0,0]
    return img_color

def normalize_single(symbol):
    symbol = np.copy(symbol)
    symbol /= np.max(symbol)
    rows, cols = symbol.shape
    # scale to 40x40
    inner_size = 40
    if rows > cols:
        factor = inner_size/rows
        rows = inner_size
        cols = int(round(cols*factor))
        inner = cv2.resize(symbol, (cols,rows))
    else:
        factor = inner_size/cols
        cols = inner_size
        rows = int(round(rows*factor))
        inner = cv2.resize(symbol, (cols, rows))
        
    # pad to 48x48
    outer_size = 48
    colsPadding = (int(math.ceil((outer_size-cols)/2.0)),int(math.floor((outer_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((outer_size-rows)/2.0)),int(math.floor((outer_size-rows)/2.0)))
    outer = np.pad(inner,(rowsPadding,colsPadding),'constant', constant_values=(1,1))
    
    # center the mass
    shiftx,shifty = getBestShift(outer)
    shifted = shift(outer,shiftx,shifty)
    return shifted
    
def getBestShift(img):
    inv = invert(img)
    cy,cx = ndimage.measurements.center_of_mass(inv)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows), borderValue=1)
    return shifted

# simple equations
list_digits = []
for i in range(10):
    list_digits.append(listdir(read_path+"/"+str(i)))
list_int = listdir(read_path+"/int")  #added 21/05 -v
list_d = listdir(read_path+"/d")  #added 21/05 -v
list_plus = listdir(read_path+"/+")
list_minus = listdir(read_path+"/-")

#commented 22/05 -p
#list_lt = listdir(read_path+"/lt")
#list_gt= listdir(read_path+"/gt")
#list_leq = listdir(read_path+"/leq")
#list_geq = listdir(read_path+"/geq")
#list_neq = listdir(read_path+"/neq")
#list_equal = listdir(read_path+"/=")
list_sin= listdir(read_path+"/sin") #added 22/05 -p
list_cos= listdir(read_path+"/cos") #added 22/05 -p
list_tan= listdir(read_path+"/tan") #added 22/05 -p
#list_alpha = listdir(read_path+"/alpha")
#list_beta = listdir(read_path+"/beta")
#list_lambda = listdir(read_path+"/lambda")
list_x = listdir(read_path+"/x")
#list_y = listdir(read_path+"/y")

if train: 
    for i in range(10):
        list_digits[i] = list_digits[i][len(list_digits[i])//4:]
    list_int = list_int[len(list_int)//4:]      #added 21/05 -v
    list_d = list_d[len(list_d)//4:]      #added 21/05 -v
    list_plus = list_plus[len(list_plus)//4:]
    list_minus = list_minus[len(list_minus)//4:]
    #commented 22/05 -p
    #list_lt = list_lt[len(list_lt)//4:]
    #list_gt = list_gt[len(list_gt)//4:]
    #list_leq = list_leq[len(list_leq)//4:]
    #list_geq = list_geq[len(list_geq)//4:]
    #list_neq = list_neq[len(list_neq)//4:]
    #list_equal = list_equal[len(list_equal)//4:]
    list_sin = list_sin[len(list_sin)//4:]  #added 22/05 -p
    list_cos = list_cos[len(list_cos)//4:]  #added 22/05 -p
    list_tan = list_tan[len(list_tan)//4:]  #added 22/05 -p
    #list_alpha = list_alpha[len(list_alpha)//4:]
    #list_beta = list_beta[len(list_beta)//4:]
    #list_lambda = list_lambda[len(list_lambda)//4:]
    list_x = list_x[len(list_x)//4:]
    #list_y = list_y[len(list_y)//4:]
else:
    for i in range(10):
        list_digits[i] = list_digits[i][:len(list_digits[i])//4]
    list_int = list_int[:len(list_int)//4]      #added 21/05 -v  
    list_d = list_d[:len(list_d)//4]      #added 21/05 -v
    list_plus = list_plus[:len(list_plus)//4]
    list_minus = list_minus[:len(list_minus)//4]
    #commented 22/05 -p
    #list_lt = list_lt[:len(list_lt)//4]
    #list_gt = list_gt[:len(list_gt)//4]
    #list_leq = list_leq[:len(list_leq)//4]
    #list_geq = list_geq[:len(list_geq)//4]
    #list_neq = list_neq[:len(list_neq)//4]
    #list_equal = list_equal[:len(list_equal)//4]
    list_sin = list_sin[:len(list_sin)//4]  #added 22/05 -p
    list_cos = list_cos[:len(list_sin)//4]  #added 22/05 -p
    list_tan = list_tan[:len(list_sin)//4]  #added 22/05 -p
    #list_alpha = list_alpha[:len(list_alpha)//4]
    #list_beta = list_beta[:len(list_beta)//4]
    #list_lambda = list_lambda[:len(list_lambda)//4]
    list_x = list_x[:len(list_x)//4]
    #list_y = list_y[:len(list_y)//4]

list_mid = [list_minus,list_plus]
#commented 22/05 -p
#list_end = [list_lt,list_gt,list_leq,list_geq,list_neq,list_equal]  
list_end = [list_sin,list_cos,list_tan]  #added 22/05 -p
list_end_str=["sin","cos","tan"] #added 22/05 -v
#commented 22/05 -p
#list_variables = [list_alpha,list_beta,list_lambda,list_x,list_y]
list_variables = [list_x]  #added 22/05 -p
var_names = ["x"]
    
# os.mkdir(write_path_add)
bounding_boxes = []
for i in range(50):
    random_name = str(np.random.randint(1,99999))
    img = np.zeros((140,60*(4+4+12)))
    ## trignometrsing digits :P
    random_trig_1 =  np.random.randint(1,3)  #used to decide if digit is a number or a trig func added 22/05 -v
    random_trig_2 =  np.random.randint(1,3)  #used to decide if digit is a number or a trig func added 22/05 -v
    rand_num_1=0
    if random_trig_1 ==1:
        rand_num_1 = np.random.randint(1,1000)
        rand_num_1_lat= str( rand_num_1)
    else:
        rand_num_1 = list_end_str[np.random.randint(0,3)]
        rand_num_1_lat= str( rand_num_1 + "x")
    if random_trig_2 ==1:
        rand_num_2 = np.random.randint(1,1000)
        rand_num_2_lat= str( rand_num_2)
    else:
        rand_num_2 = list_end_str[np.random.randint(0,3)]
        rand_num_2_lat= str( rand_num_2 + "x")
        ## 22/05 -v
        
    mid = np.random.randint(2)
    if mid:
        mid_str = "+"
       # result = rand_num_1+rand_num_2
    else:
        mid_str = "-"
       # result = rand_num_1-rand_num_2
    result = np.random.randint(1,1000)  ##the above lines have been commented by me 
                                        #to remove errors when rand nos are not actually nos
                                        #result is now a randomly generated number 
    
    result_type = np.random.randint(3)
    if result_type == 0:
        #end_str = " #lt "
        end_str = " #sin"
        result += np.random.randint(100)
    elif result_type == 1:
        #end_str = " #gt "
        end_str = " #cos "
        result -= np.random.randint(100)
    elif result_type == 2:
        #end_str = " #leq "
        end_str = " #tan "
        result += np.random.randint(100)
    elif result_type == 3:
        #end_str = " #geq "
        end_str = " #sin"
        result -= np.random.randint(100)
    elif result_type == 4:
        #end_str = " #neq "
        end_str = " #cos "
        result += 1+np.random.randint(100)
    else:
        end_str = " = "
    if end_str != " = ":
        var_type = np.random.randint(len(list_variables))
        var = var_names[var_type]
        exp = "^"+str(np.random.randint(2,5))
    else:
        var = ""
        exp = ""
    
    rand_num_1_str = str(rand_num_1)
    rand_num_2_str = str(rand_num_2)
    result_str = str(result)
    num_strs = [rand_num_1_str,rand_num_2_str,result_str]
    filename = rand_num_1_lat+var+exp+mid_str+rand_num_2_lat+end_str+result_str+"_"+random_name+".jpg"
    print("Filename: ", filename)
    bounding_box = [{'filename': filename}]
    padding = 5
    class_names = []
    
    ##adding the integral to image
    img, padding, new_bounding_box = add_symbol_to_image(img,'int',list_int,
                                                                     padding,70,75)
    bounding_box.append(new_bounding_box)
    class_names.append('int')
    #added 21/05 - v
    
    for k in range(3):
        if(num_strs[k]=="sin" or num_strs[k]=="cos" or num_strs[k]=="tan" ):
            if num_strs[k]=="sin":
                result_typevik=0
            elif(num_strs[k]=="cos"):
                result_typevik=1
            else:
                result_typevik=2
            img, padding, new_bounding_box = add_symbol_to_image(img,num_strs[k],list_end[result_typevik],
                                                                     padding,39,54)
            bounding_box.append(new_bounding_box)
            class_names.append(num_strs[k])
            #adding x to sin
            img, padding, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,50,55)
            bounding_box.append(new_bounding_box)
            class_names.append('x')
            img, padding, new_bounding_box = add_symbol_to_image(img,mid_str,list_mid[mid],
                                                                     padding,39,54)
            bounding_box.append(new_bounding_box)
            class_names.append(mid_str)
            continue
         ## sin cos tan code ends here 
        for j in range(len(num_strs[k])):
            if j == 0 and num_strs[k][0] == '-':
                img, padding, new_bounding_box = add_symbol_to_image(img,'-',list_minus,padding,39,45)
                bounding_box.append(new_bounding_box)
                class_names.append('-')
            else:
                digit = int(num_strs[k][j])
                img, padding, new_bounding_box = add_symbol_to_image(img,str(digit),list_digits[digit],
                                                                     padding,55,60)
                bounding_box.append(new_bounding_box)
                class_names.append(str(digit))
        if k == 0:
            if var != "":
                var_str_crop = var.strip()
                var_str_crop = var_str_crop.replace("#","")
                img, padding, new_bounding_box = add_symbol_to_image(img,var_str_crop,list_variables[var_type],
                                                                     padding,40,45,bsmall=True)
                bounding_box.append(new_bounding_box)
                class_names.append(var_str_crop)
                
                pdigit = int(exp[1:])
                img, padding, new_bounding_box = add_symbol_to_image(img,str(pdigit),list_digits[pdigit],
                                                                     padding,35,40,bpower=True)
                bounding_box.append(new_bounding_box)
                class_names.append(exp[1:])
                
            img, padding, new_bounding_box = add_symbol_to_image(img,mid_str,list_mid[mid],
                                                                     padding,39,54)
            bounding_box.append(new_bounding_box)
            class_names.append(mid_str)
        elif k == 1:
            end_str_crop = end_str.strip()
            end_str_crop = end_str_crop.replace("#","")
            img, padding, new_bounding_box = add_symbol_to_image(img,end_str_crop,list_end[result_type],
                                                                     padding,39,54)
            bounding_box.append(new_bounding_box)
            class_names.append(end_str_crop)
            #adding x to sin
            img, padding, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,50,55)
            bounding_box.append(new_bounding_box)
            class_names.append('x')
            #added 22/05 -p
            
    ##adding the dx to image
    img, padding, new_bounding_box = add_symbol_to_image(img,'d',list_d,
                                                                     padding,50,55)
    bounding_box.append(new_bounding_box)
    class_names.append('d')
    img, padding, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,50,55)
    bounding_box.append(new_bounding_box)
    class_names.append('x')
    
    #added 21/05 - v        
            
    bounding_boxes.append(bounding_box)
    img = invert(img)+254
#     plt.figure(figsize=(20,10))
#     plt.imshow(img, cmap="gray")
#     plt.show()
    for bb,cname in zip(bounding_box[1:],class_names):
        xmin, xmax = bb['xmin'], bb['xmax']
        ymin, ymax = bb['ymin'], bb['ymax']
    
        normed = normalize_single(img[ymin:ymax+1,xmin:xmax+1])
        r = np.random.randint(9999)
        io.imsave(write_single_path+"/"+cname+"/"+cname+"_"+str(r)+".jpg", normed)
    
    io.imsave(formula_path+"/"+filename, img/255)
    print("Finished: ", i)
    
    
    

# fractions
list_digits = []
for i in range(10):
    list_digits.append(listdir(read_path+"/"+str(i)))
list_plus = listdir(read_path+"/+")
list_minus = listdir(read_path+"/-")
#commented 22/05 -p
#list_leq = listdir(read_path+"/leq")
#list_geq = listdir(read_path+"/geq")
#list_neq = listdir(read_path+"/neq")
#list_equal = listdir(read_path+"/=")
list_sin = listdir(read_path+"/sin")  #added 22/05 -p
list_cos = listdir(read_path+"/cos")  #added 22/05 -p
list_tan = listdir(read_path+"/tan")  #added 22/05 -p

#list_alpha = listdir(read_path+"/alpha")
#list_beta = listdir(read_path+"/beta")
#list_lambda = listdir(read_path+"/lambda")
list_x = listdir(read_path+"/x")
#list_y = listdir(read_path+"/y")

if train: 
    for i in range(10):
        list_digits[i] = list_digits[i][len(list_digits[i])//4:]
    list_int = list_int[len(list_int)//4:]      #added 21/05 -vikram
    list_plus = list_plus[len(list_plus)//4:]
    list_minus = list_minus[len(list_minus)//4:]
    #commented 22/05 -p
    #list_leq = list_leq[len(list_leq)//4:]
    #list_geq = list_geq[len(list_geq)//4:]
    #list_neq = list_neq[len(list_neq)//4:]
    #list_equal = list_equal[len(list_equal)//4:]
    list_sin = list_sin[len(list_sin)//4:]   #added 22/05 -p
    list_cos = list_cos[len(list_cos)//4:]   #added 22/05 -p
    list_tan = list_tan[len(list_tan)//4:]   #added 22/05 -p
    #list_alpha = list_alpha[len(list_alpha)//4:]
    #list_beta = list_beta[len(list_beta)//4:]
    #list_lambda = list_lambda[len(list_lambda)//4:]
    list_x = list_x[len(list_x)//4:]
    #list_y = list_y[len(list_y)//4:]
else:
    for i in range(10):
        list_digits[i] = list_digits[i][:len(list_digits[i])//4]
    list_int = list_int[:len(list_int)//4]      #added 21/05 -vikram 
    list_plus = list_plus[:len(list_plus)//4]
    list_minus = list_minus[:len(list_minus)//4]
    #commented 22/05 -p
    #list_leq = list_leq[:len(list_leq)//4]
    #list_geq = list_geq[:len(list_geq)//4]
    #list_neq = list_neq[:len(list_neq)//4]
    #list_equal = list_equal[:len(list_equal)//4]
    list_sin = list_sin[:len(list_sin)//4]   #added 22/05 -p
    list_cos = list_cos[:len(list_cos)//4]   #added 22/05 -p
    list_tan = list_tan[:len(list_tan)//4]   #added 22/05 -p
    #list_alpha = list_alpha[:len(list_alpha)//4]
    #list_beta = list_beta[:len(list_beta)//4]
    #list_lambda = list_lambda[:len(list_lambda)//4]
    list_x = list_x[:len(list_x)//4]
    #list_y = list_y[:len(list_y)//4]

list_mid = [list_minus,list_plus]
#commented 22/05 -p
#list_end = [list_leq,list_geq,list_neq,list_equal]
list_end = [list_sin,list_cos,list_tan]  #added 22/05 -p
#list_variables = [list_alpha,list_beta,list_lambda,list_x,list_y]
list_variables = [list_x]  #added 22/05 -p
var_names = ["x"]
    
# os.mkdir(write_path_add)
bounding_boxes = []
for i in range(50):
    random_name = str(np.random.randint(1,99999))
    img = np.zeros((200,75*(4+4+10)))
    ## trignometrsing digits :P
    random_trig_1 =  np.random.randint(1,3)  #used to decide if digit is a number or a trig func added 22/05 -v
    random_trig_2 =  np.random.randint(1,3)  #used to decide if digit is a number or a trig func added 22/05 -v
    #random_trig_3 =  np.random.randint(1,3)  #used to decide if digit is a number or a trig func added 22/05 -v
    rand_num_1=0
    if random_trig_1 ==1:
        rand_num_1 = np.random.randint(1,1000)
        rand_num_1_lat= str( rand_num_1)
    else:
        rand_num_1 = list_end_str[np.random.randint(0,3)]
        rand_num_1_lat= str( rand_num_1 + "x")
    if random_trig_2 ==1:
        rand_num_2 = np.random.randint(1,1000)
        rand_num_2_lat= str( rand_num_2 )
    else:
        rand_num_2 = list_end_str[np.random.randint(0,3)]
        rand_num_2_lat= str( rand_num_2 + "x")
        ## 22/05 -v
    #if random_trig_3 ==1:
    #    rand_num_1_1 = np.random.randint(1,1000)
    #else:
    #    rand_num_1_1 = list_end_str[np.random.randint(0,3)]
        ## 22/05 -v    
    #rand_num_1 = np.random.randint(1,1000)
    rand_num_1_1 = np.random.randint(1,1000)
    #rand_num_2 = np.random.randint(1,1000)
    mid = np.random.randint(2)
    if mid:
        mid_str = "+"
        #result = rand_num_1+rand_num_2
    else:
        mid_str = "-"
        #result = rand_num_1-rand_num_2
    result = np.random.randint(1,1000) ##result doesnt have any significance
    result_type = np.random.randint(3)
    if result_type == 0:
        #end_str = " #leq "
        end_str = " #sin"
        result += np.random.randint(100)
    elif result_type == 1:
        #end_str = " #geq "
        end_str = " #cos "
        result -= np.random.randint(100)
    elif result_type == 2:
        #end_str = " #neq "
        end_str = " #tan "
        result += 1+np.random.randint(100)
    else:
        end_str = " = "
    if end_str != " = ":
        var_type = np.random.randint(len(list_variables))
        var = var_names[var_type]
        exp = "^"+str(np.random.randint(2,5))
    else:
        var = ""
        exp = ""
    
    rand_num_1_str = str(rand_num_1)
    rand_num_1_1_str = str(rand_num_1_1)
    rand_num_2_str = str(rand_num_2)
    result_str = str(result)
    num_strs = [rand_num_1_str,rand_num_1_1_str,rand_num_2_str,result_str]
    filename = "#frac{"+rand_num_1_lat+"}{"+rand_num_1_1_str+"}"+var+exp+mid_str+rand_num_2_lat+end_str+result_str+"_"+random_name+".jpg"
    print("Filename: ", filename)
    bounding_box = [{'filename': filename}]
    padding = 5
    padding_den = 5
    start_padding = 5
    class_names = []
    
    ##adding the integral to image
    digit = 'int'
    img, padding, new_bounding_box = add_symbol_to_image(img,'int',list_int,
                                                                     padding,90,95)
    bounding_box.append(new_bounding_box)
    class_names.append('int')
    #added 21/05 - v
    
    for k in range(len(num_strs)):        
        if k == 0:                
            if(num_strs[k]=="sin" or num_strs[k]=="cos" or num_strs[k]=="tan" ):
                    if num_strs[k]=="sin":
                        result_typevik=0
                    elif(num_strs[k]=="cos"):
                        result_typevik=1
                    else:
                        result_typevik=2
                    img, padding, new_bounding_box = add_symbol_to_image(img,num_strs[k],list_end[result_typevik],
                                                                     padding,39,54,bnom= True)
                    bounding_box.append(new_bounding_box)
                    class_names.append(num_strs[k])
                    #adding x to sin
                    img, padding, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,50,55,bnom=True)
                    bounding_box.append(new_bounding_box)
                    class_names.append('x')
                    continue
        elif k == 1:                
            if(num_strs[k]=="sin" or num_strs[k]=="cos" or num_strs[k]=="tan" ):
                    if num_strs[k]=="sin":
                        result_typevik=0
                    elif(num_strs[k]=="cos"):
                        result_typevik=1
                    else:
                        result_typevik=2
                    img, padding_den, new_bounding_box = add_symbol_to_image(img,num_strs[k],list_end[result_typevik],
                                                                     padding,39,54,bden=True)
                    bounding_box.append(new_bounding_box)
                    class_names.append(num_strs[k])
                    #adding x to sin
                    img, padding_den, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,50,55,bden=True)
                    bounding_box.append(new_bounding_box)
                    class_names.append('x')
                    img, padding, new_bounding_box = add_symbol_to_image(img,mid_str,list_mid[mid],
                                                                         padding,39,54)
                    bounding_box.append(new_bounding_box)
                    class_names.append(mid_str)
                    continue
        else:
            if(num_strs[k]=="sin" or num_strs[k]=="cos" or num_strs[k]=="tan" ):
                if num_strs[k]=="sin":
                    result_typevik=0
                elif(num_strs[k]=="cos"):
                    result_typevik=1
                else:
                    result_typevik=2
                img, padding, new_bounding_box = add_symbol_to_image(img,num_strs[k],list_end[result_typevik],
                                                                     padding,39,54)
                bounding_box.append(new_bounding_box)
                class_names.append(num_strs[k])
                #adding x to sin
                img, padding, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,35,39)
                bounding_box.append(new_bounding_box)
                class_names.append('x')
                img, padding, new_bounding_box = add_symbol_to_image(img,mid_str,list_mid[mid],
                                                                         padding,39,54)
                bounding_box.append(new_bounding_box)
                class_names.append(mid_str)
                continue
        
        
        for j in range(len(num_strs[k])):   
            if j == 0 and num_strs[k][0] == '-':
                img, padding, new_bounding_box = add_symbol_to_image(img,'-',list_minus,padding,39,45)
                bounding_box.append(new_bounding_box)
                class_names.append('-')
            elif k == 0:
                
                digit = int(num_strs[k][j])
                img, padding, new_bounding_box = add_symbol_to_image(img,str(digit),list_digits[digit],
                                                                     padding,40,45,bnom=True)
                bounding_box.append(new_bounding_box)
                class_names.append(str(digit))
            elif k==1:
                
                digit = int(num_strs[k][j])
                img, padding_den, new_bounding_box = add_symbol_to_image(img,str(digit),list_digits[digit],
                                                                     padding_den,40,45,bden=True)
                bounding_box.append(new_bounding_box)
                class_names.append(str(digit))
            else:               
                digit = int(num_strs[k][j])
                img, padding, new_bounding_box = add_symbol_to_image(img,str(digit),list_digits[digit],
                                                                     padding,55,60)
                bounding_box.append(new_bounding_box)
                class_names.append(str(digit))
        if k == 1:
            mpad = padding if padding > padding_den else padding_den
            img, padding, new_bounding_box = add_symbol_to_image(img,'-',list_minus,start_padding,39,45,
                                                                 width=mpad-start_padding)
            bounding_box.append(new_bounding_box)
            class_names.append('-')
        if k == 1:
            if var != "":
                var_str_crop = var.strip()
                var_str_crop = var_str_crop.replace("#","")
                img, padding, new_bounding_box = add_symbol_to_image(img,var_str_crop,list_variables[var_type],
                                                                     padding,40,45,bsmall=True)
                bounding_box.append(new_bounding_box)
                class_names.append(var_str_crop)
                
                pdigit = int(exp[1:])
                img, padding, new_bounding_box = add_symbol_to_image(img,str(pdigit),list_digits[pdigit],
                                                                     padding,35,40,bpower=True)
                bounding_box.append(new_bounding_box)
                class_names.append(exp[1:])
                
            img, padding, new_bounding_box = add_symbol_to_image(img,mid_str,list_mid[mid],
                                                                     padding,39,54)
            bounding_box.append(new_bounding_box)
            class_names.append(mid_str)
        elif k == 2:
            end_str_crop = end_str.strip()
            end_str_crop = end_str_crop.replace("#","")
            img, padding, new_bounding_box = add_symbol_to_image(img,end_str_crop,list_end[result_type],
                                                                     padding,39,54)
            bounding_box.append(new_bounding_box)
            class_names.append(end_str_crop)
            ##adding the dx to image
    img, padding, new_bounding_box = add_symbol_to_image(img,'d',list_d,
                                                                     padding,50,55)
    bounding_box.append(new_bounding_box)
    class_names.append('d')
    img, padding, new_bounding_box = add_symbol_to_image(img,'x',list_x,
                                                                     padding,50,55)
    bounding_box.append(new_bounding_box)
    class_names.append('x')
    
    #added 21/05 - v 
    bounding_boxes.append(bounding_box)
    img = invert(img)+254
#     plt.figure(figsize=(20,10))
#     plt.imshow(img, cmap="gray")
#     plt.show()
    """
    for bb,cname in zip(bounding_box[1:],class_names):
        xmin, xmax = bb['xmin'], bb['xmax']
        ymin, ymax = bb['ymin'], bb['ymax']
    
        normed = normalize_single(img[ymin:ymax+1,xmin:xmax+1])
        r = np.random.randint(9999)
        io.imsave(write_single_path+"/"+cname+"/"+cname+"_"+str(r)+".jpg", normed)
    """
    io.imsave(formula_path+"/"+filename, img/255)
    print("Finished: ", i)

