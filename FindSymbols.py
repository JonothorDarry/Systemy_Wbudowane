import numpy as np
import cv2 as cv
import time
import random
import math
import sys
import os
from photo import *

data_path = 'data/'
symbols_path = os.path.abspath(os.path.dirname(__file__))+'/symbols/'


symbol_paths = ['Note-Heads/1.png', 'Note-Heads/3.png', 'Note-Heads/2.png']
symbols = []
for s in symbol_paths:
    symbol = cv.imread(symbols_path + s, cv.IMREAD_GRAYSCALE)
    #_, symbol = cv.threshold(symbol, 127, 255, cv.THRESH_BINARY_INV)
    symbols.append(symbol)

def plot_gray(img):
    plt.imshow(img, vmin = 0, vmax = 255, cmap = 'gray')
    
def plot_color(img):
    plt.imshow(img)
    
def plot_in_window(img):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def resize(img, max_resolution = 1500):
    rs_factor = max_resolution / max(img.shape)
    if rs_factor >= 1:
        return img
    return cv.resize(img, None, fx = rs_factor, fy = rs_factor, interpolation = cv.INTER_AREA)

def put_text(img, text, coords):
    return cv.putText(img, text, coords, cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

def grave(img, biw=0):
    fig, ax=plt.subplots(1,1,figsize=(22,22))
    if (biw==1):
        ax.imshow(img, cmap='Greys',  interpolation='nearest')
    else:
        ax.imshow(img)
    plt.show()
    return

def decorate(i):
    print()
    print('#'*30+'   '+str(i)+'   '+'#'*30)
    print()


Hypers={'Iterate_over':(26, 0), 'Binarization_conn':65, 'Binarization_blur':(6,1), 'Dt':(40, 80), 'Precise':1}


kernel=[0]*10
#kernele do grafiki
kernel[1]=np.asarray([[-1, -1, -1], [-1,8,-1], [-1, -1, -1]])
kernel[2]=np.asarray([[0, 1, 0], [1,-4,1], [0, 1, 0]])
kernel[3]=np.asarray([[0, 1, 0], [1,-5,1], [0, 1, 0]])
kernel[4]=(1/9)*np.asarray([[1, 1, 1], [1,1,1], [1, 1, 1]])
kernel[5]=(1/16)*np.asarray([[1, 2, 1], [2,4,2], [1, 2, 1]])
kernel[6]=(1/256)*np.asarray([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
dm=(5,5)
kernel[7]=np.asarray([[1 for j in range(dm[0])] for i in range(dm[1])])
kernel[8]=np.asarray([[0, -1, 0], [-1,5,-1], [0, -1, 0]])

def binarization(bwimg):
    edges = cv.Canny(bwimg,50,150,apertureSize = 3)
    edges2=cv.filter2D(bwimg, -1, kernel[1])
    edges2=cv.Canny(edges2,50,150,apertureSize = 3)
    bwimg=cv.adaptiveThreshold(bwimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, Hypers['Binarization_conn'], 1)
    
    if (Hypers['Binarization_blur'][0]>0):
        bw=cv.filter2D(edges, -1, kernel[7])
        dw=cv.filter2D(edges2, -1, kernel[7])
        for jj in range(Hypers['Binarization_blur'][0]-1):
            bw=cv.filter2D(bw, -1, kernel[7])
            dw=cv.filter2D(dw, -1, kernel[7])
        #grave(dw)
        if (Hypers['Precise']==1):
            bwimg[dw<Hypers['Binarization_blur'][1]]=255
        else:
            bwimg[bw<Hypers['Binarization_blur'][1]]=255
    
    return bwimg


def rotate_image(img, fimg):
    #Useless
    minLineLength = 100
    maxLineGap = 10
    
    kern=(1/256)*np.asarray([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
    #img2=cv.filter2D(img, -1, kernel[2])
    img2=fimg.copy()
    #img2=cv.filter2D(img2, -1, kernel[7])
    #img2=cv.filter2D(img2, -1, kernel[7])
    #img2=cv.filter2D(img2, -1, kernel[7])
    edges = cv.Canny(img2,50,150,apertureSize = 3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
    #print(lines)
    if (lines is None):
        lines=[]
    cosa=[0]*len(lines)
    lenny=[0]*len(lines)
    for i, iv in enumerate(lines):
        x1,y1,x2,y2=iv[0]
        if (x2<x1 or (x2==x1 and y2<y1)):
            f1, f2=x1, y1
            x1, y1=x2, y2
            x2, y2=x1, y1
        
        dx, dy=(x2-x1), (y2-y1)
        
        if (dx*dx+dy*dy>0):
            cosa[i]=dy/math.sqrt(dy*dy+dx*dx)
        else:
            cosa[i]=0
        lenny[i]=math.sqrt(dy*dy+dx*dx)
        #cv.line(img,(x1,y1),(x2,y2),(255,0,0),1)
    
    x=0
    blyat=[0]*210
    for i in range(len(cosa)):
        s=math.floor(cosa[i]*100)+100
        blyat[s]+=lenny[i]
        if (blyat[s]>blyat[x]):
            x=s
    x=(x-100)/100
    
        
    angle=90-math.acos(x)*(180/math.pi)
    
    h, w = img.shape[:2]
    image_center = (w/2, h/2)
    
    
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((h*abs(sin)) + (w*abs(cos)))
    bound_h = int((h*abs(cos)) + (w*abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv.warpAffine(img, rotation_mat, (bound_w, bound_h), borderValue=255)
    rotated_org = cv.warpAffine(fimg, rotation_mat, (bound_w, bound_h), borderValue=255)
    return rotated_mat, rotated_org

C=100000
par=list(range(C))
rank=[0]*C
great_dis=[0]*C

def make_set(i):
    par[i]=i
    rank[i]=0
    great_dis[i]=0

def find_par(a):
    if (a!=par[a]):
        par[a]=find_par(par[a])
    return par[a]


def union(x, y, v):
    a=find_par(x)
    b=find_par(y)
    
    great_dis[a]=max(great_dis[a], great_dis[b], v)
    great_dis[b]=max(great_dis[a], great_dis[b], v)
    if (rank[a]<rank[b]):
        par[a]=b
    else:
        par[b]=a
    if (rank[a]==rank[b]):
        rank[a]+=1

def line5finder(sol, img):
    dp=[0]*len(sol)
    kenose=[0]*len(sol)
    tv=[0]*len(sol)
    dtrl, dtrr=Hypers['Dt']
    for i, x in enumerate(sol):
        make_set(i)
        if (i>0):
            dp[i]=sol[i][0]-sol[i-1][0]
            tv[i]=i
            
    vv=zip(dp,tv)
    vv=sorted(vv, key=lambda x:x[0])
    dp, tv=tuple(zip(*list(vv)))
    for i, x in enumerate(dp):
        if (dp[i]<6 or ((2*great_dis[find_par(tv[i]-1)]>dp[i] or great_dis[find_par(tv[i]-1)]<5) and (2*great_dis[find_par(tv[i])]>dp[i] or great_dis[find_par(tv[i])]<5))):
            union(tv[i], tv[i]-1, dp[i])
            
    
    cl=(1, 0, 0)
    chan=0
    outer=[0]*len(sol)
    wynne=[]
    for i in range(len(sol)):
        if (i>0 and find_par(i)!=find_par(i-1)):
            chan=chan+1
        outer[chan]+=1
        
    
    #fig, ax=plt.subplots(1,1,figsize=(18,18))
    for i in range(len(sol)):
        if (i>0 and find_par(i)!=find_par(i-1)):
            cl=(random.random(), 0, random.random())
        vb=sol[i][4]
        #ax.plot([vb[1][1], vb[-1][1]], [sum(list(zip(*vb))[0][dtrl:dtrr])/(dtrr-dtrl), sum(list(zip(*vb))[0][-dtrr:-dtrl])/(dtrr-dtrl)], color=cl)

    #ax.imshow(img, cmap='Greys',  interpolation='nearest')
    #plt.show()
    
    chan=0
    xlm, xrm, yb, ye=100000, 0, 0, 0
    vb=sol[0][4]
    for i, x in enumerate(sol):
        if (i>0 and find_par(i)!=find_par(i-1)):
            ye=sol[i-1][0]
            ve=sol[i-1][4]
            if (outer[chan]>=3 and outer[chan]<=15):
                #wynne teraz zawiera 4 listy, kolejno: [xbl, xbr], [ybl, ybr], [xel, xer], [yel, yer]
                #x - ixowa, y - ygrekowa, b - 1. linia, e - ostatnia linia, l - lewo, r - prawo
                wynne.append(([vb[1][1], vb[-1][1]], [sum(list(zip(*vb))[0][dtrl:dtrr])/(dtrr-dtrl), sum(list(zip(*vb))[0][-dtrr:-dtrl])/(dtrr-dtrl)], [ve[1][1], ve[-1][1]], [sum(list(zip(*ve))[0][dtrl:dtrr])/(dtrr-dtrl), sum(list(zip(*ve))[0][-dtrr:-dtrl])/(dtrr-dtrl)]))
            
            chan=chan+1
            yb=x[0]
            vb=x[4]
    """
    fig, ax=plt.subplots(1,1,figsize=(18,18))
    for i, x in enumerate(wynne):
        ax.plot(x[0], x[1], color=cl)
        ax.plot(x[2], x[3], color=cl)
        cl=(random.random(), 0, random.random())
    ax.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.show()
    """
    return wynne


def remove_staff_lines(input_img):
    t1=time.time()
    #decorate(i)
    imgb=input_img.copy()
    #grave(imgb, 1)
    shorig=imgb.shape
    
    ###BINARIZATION
    imv=binarization(imgb.copy())
    #Rotacja po linii - czasem jeszcze nie działa
    img2, img_original =rotate_image(imv, imgb)
    img2=cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, Hypers['Binarization_conn'], 1)
    #grave(img2, 1)
    
    ss=np.zeros((img2.shape[0]+20, img2.shape[1]+20), dtype=str(img2.dtype))
    ss[10:-10,10:-10]=img2
    img2=ss.copy()
    
    # Changes - padding original image
    padded = np.zeros((img_original.shape[0]+20, img_original.shape[1]+20), dtype=str(img_original.dtype))
    padded[10:-10,10:-10] = img_original
    
    
    ###DETECTION 
    sol=findlinez(img2, shorig)
    #sol - lista tupli - 1-wszy to uśrednione miejsce linii pięciolinii, 2-gi to grubość linii
    wynne=line5finder(sol, img2)
    
    #grave(img2, 1)
    t2=time.time()
    #print(t2-t1)
    
    return ~img2, padded, wynne


class Note:
    def __init__(self, x1, y1, x2, y2):
        self.bbox = ((x1, y1), (x2, y2))
        self.center = ((x1 + x2)//2, (y1 + y2)//2)

    def __key(self):
        return self.bbox

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Note):
            return self.__key() == other.__key()
        return NotImplemented

    def contains(self, other):
        return (self.bbox[0][0] <= other.center[0] <= self.bbox[1][0]
                and self.bbox[0][1] <= other.center[1] <= self.bbox[1][1])


def get_staff_height(staff_lines):
    heights = [staff[3][i] - staff[1][i] for staff in staff_lines for i in range(2)]
    return np.median(heights)

def get_staff_left_right(staff_lines):
    x_left = [staff[i][0] for staff in staff_lines for i in (0, 2)]
    x_right = [staff[i][1] for staff in staff_lines for i in (0, 2)]
    return np.median(x_left), np.median(x_right)

def get_staff_x(x1, x2, median, staff_length):
    th = staff_length/3
    if abs(median - x1) > th and abs(median - x2) > th:
        return median, median
    if abs(x1 - median) < abs(x2 - median):
        return x1, x1
    return x2, x2

def get_stem_end(img, x, y, up):
    step = 2 * up - 1
    limit = 3
    last_white_y = y
    while abs(y - last_white_y) <= limit:
        y = y - step
        if y < 0 or y >= img.shape[0]:
            break
        if img[y, x] == 255:
            last_white_y = y
    return last_white_y - step

TREBLE, BASS, UNKNOWN = 1, 2, -1
def get_note_pitch(position, clef):
    pitch = ['C', 'D', 'E', 'F', 'G', 'A', 'H']
    if clef == TREBLE:
        return pitch[(position + 2) % len(pitch)]
    elif clef == BASS:
        return pitch[(position + 4) % len(pitch)]
    return 'C'

def match_template(img, template, th):
    result = cv.matchTemplate(img, template, method = cv.TM_SQDIFF_NORMED)

    # Remove duplicate detections
    h, w = template.shape
    noteheads = np.nonzero(result < th)
    detected_noteheads = set()
    for x, y in zip(noteheads[1], noteheads[0]):
        x = max(0, x - w//2)
        y = max(0, y - h//2)
        neighborhood = result[y:(y + h), x:(x + w)]
        note_y, note_x = np.unravel_index(np.argmin(neighborhood), neighborhood.shape)
        note_x = note_x + x
        note_y = note_y + y
        detected_noteheads.add(Note(note_x, note_y, note_x + w, note_y + h))
    return detected_noteheads

def add_noteheads(noteheads, detected_noteheads):
    # Add notes that are not duplicates
    for note in detected_noteheads:
        is_duplicate = False
        for other_note in noteheads:
            if note.contains(other_note):
                is_duplicate = True
                break
        if not is_duplicate:
            noteheads.append(note)
    return noteheads

def detect_notes(img, img_original, staff_lines):

    img_color = cv.cvtColor(img_original, cv.COLOR_GRAY2RGB)
    # Resize symbols
    staff_height = get_staff_height(staff_lines)
    note_height = staff_height/4 * 1.25

    rs_factor = note_height / symbols[0].shape[0]
    note_head = cv.resize(~symbols[0], None, fx = rs_factor, fy = rs_factor, interpolation = cv.INTER_AREA)

    rs_factor = note_height * 1.1 / symbols[1].shape[0]
    note_head_half = cv.resize(~symbols[1], None, fx = rs_factor, fy = rs_factor, interpolation = cv.INTER_AREA)

    rs_factor = note_height * 1.1 / symbols[2].shape[0]
    note_head_whole = cv.resize(~symbols[2], None, fx = rs_factor, fy = rs_factor, interpolation = cv.INTER_AREA)

    # Cleanup of staff lines
    x_left, x_right = get_staff_left_right(staff_lines)
    staff_length = x_right - x_left
    new_staff_lines = []
    for staff in staff_lines:
        y_low = (staff[3][0] + staff[3][1])/2
        y_up = (staff[1][0] + staff[1][1])/2
        staff_h = y_low - y_up
        if staff_h > 3 * staff_height and staff[1][0] <= 10:
            continue
        if staff_h > 2 * staff_height or staff_h < staff_height/2:
            if abs(staff[0][1] - staff[0][0] - staff_length) < abs(staff[2][1] - staff[2][0] - staff_length):
                staff[3][0] = staff[1][0] + staff_height
                staff[3][1] = staff[1][1] + staff_height
            else:
                staff[1][0] = staff[3][0] - staff_height
                staff[1][1] = staff[3][1] - staff_height

        staff[0][0], staff[2][0] = get_staff_x(staff[0][0], staff[2][0], x_left, staff_length)
        staff[0][1], staff[2][1] = get_staff_x(staff[0][1], staff[2][1], x_right, staff_length)
        x_begin = (staff[0][0] + staff[2][0]) / 2
        y_center = (staff[1][0] + staff[3][0]) / 2
        new_staff_lines.append(staff)
        img_color = cv.line(img_color, (int(staff[0][0]), int(staff[1][0])),
                            (int(staff[0][1]), int(staff[1][1])), (0, 200, 50), 2)
        img_color = cv.line(img_color, (int(staff[2][0]), int(staff[3][0])),
                            (int(staff[2][1]), int(staff[3][1])), (0, 200, 50), 2)

    staff_lines = new_staff_lines

    # Detect clefs
    clefs = []
    for staff in staff_lines:
        x_begin = int((staff[0][0] + staff[2][0]) / 2)
        y_center = int((staff[1][0] + staff[3][0]) / 2)
        staff_h = int(staff_height)
        white = []
        for x in range(x_begin, x_begin + staff_h):
            roi = img[max(y_center - staff_h, 0):y_center + staff_h, x:x + staff_h]
            white.append(np.average(roi)/255)

        clef_x = np.argmax(white) + x_begin
        clef_type = TREBLE
        th = 0.23
        if white[clef_x - x_begin] > th:
            clef_type = TREBLE
            img_color = cv.rectangle(img_color, (clef_x, y_center - staff_h),
                                     (clef_x + staff_h, y_center + staff_h), (0, 200, 50), 2)
        elif white[clef_x - x_begin] > 0.03:
            clef_type = BASS
            img_color = cv.rectangle(img_color, (clef_x, y_center - staff_h),
                                     (clef_x + staff_h, y_center + staff_h), (0, 50, 200), 2)
        clefs.append((clef_type, clef_x))


    # Detect noteheads
    detected_notes = list(match_template(img, note_head, 0.32))

    detected_half_noteheads = match_template(img, note_head_half, 0.75)
    detected_notes = add_noteheads(detected_notes, detected_half_noteheads)

    detected_whole_noteheads = match_template(img, note_head_whole, 0.52)
    detected_notes = add_noteheads(detected_notes, detected_whole_noteheads)

    detected_notes = [note for note in detected_notes
                      if x_left - staff_height < note.center[0] < x_right + staff_height]
    # Detect note's location
    for note in detected_notes:
        note.location = 100000
        note.pitch = 'C'
        for i, (staff, clef) in enumerate(zip(staff_lines, clefs)):
            y_low = (staff[3][0] + staff[3][1])/2
            y_up = (staff[1][0] + staff[1][1])/2
            bin_size = (y_low - y_up)/8
            location = int(round((y_low - note.center[1])/bin_size))
            if abs(location - 4) < abs(note.location - 4):
                note.location = location
                note.pitch = get_note_pitch(location, clef[0])
                note.staff = i

    detected_notes = [note for note in detected_notes
                      if (-4 <= note.location <= 12
                          and staff_lines[note.staff][0][0] + 2 * staff_height < note.center[0])]


    # Check if notehead is empty or not
    clahe = cv.createCLAHE(clipLimit = 30.0)
    img_eq = clahe.apply(img_original)
    for note in detected_notes:
        (x, y), (x_max, y_max) = note.bbox
        x_center, y_center = note.center
        notehead_roi = img_eq[(y_center + y)//2:(y_center + y_max)//2, (x_center + x)//2:(x_center + x_max)//2]
        avg = np.average(notehead_roi)/255
        #print(avg)
        if avg > 0.1:
            note.duration = 2
        else:
            note.duration = 4

    # Detect stems using vertical projection
    for note in detected_notes:
        (x, y), (x_max, y_max) = note.bbox
        up = img[max(y_max - int(round(staff_height)), 0):y_max, x:x_max]
        down = img[y:y + int(round(staff_height)), x:x_max]
        projection_up = cv.reduce(up, 0, cv.REDUCE_AVG)/255
        projection_down = cv.reduce(down, 0, cv.REDUCE_AVG)/255

        stem_x_up = np.argmax(projection_up) + x
        stem_x_down = np.argmax(projection_down) + x
        stem_y_up = get_stem_end(img, stem_x_up, y, up = True)
        stem_y_down = get_stem_end(img, stem_x_down, y_max, up = False)
        th = staff_height/4
        has_stem = True
        if y - stem_y_up > stem_y_down - y_max:
            if y - stem_y_up < th:
                has_stem = False
            note.is_up = True
            note.stem_end = (stem_x_up, stem_y_up)
            note.stem_bbox = ((x, stem_y_up), (x_max, y_max))
        else:
            if stem_y_down - y_max < th:
                has_stem = False
            note.is_up = False
            note.stem_end = (stem_x_down, stem_y_down)
            note.stem_bbox = ((x, y), (x_max, stem_y_down))
        if not has_stem:
            note.stem_bbox = note.bbox
            if note.duration == 2:
                note.duration = 1
            else:
                #img_color = cv.rectangle(img_color, (x, y), (x_max, y_max), (0, 0, 255), 2)
                note.duration = -1

    detected_notes = [note for note in detected_notes if note.duration >= 0]


    # Check if note is an eight note
    h, w = note_head.shape
    for note in detected_notes:
        if note.duration < 4:
            continue

        center, bbox, (stem_x, stem_y), is_up = note.center, note.stem_bbox, note.stem_end, note.is_up

        # Detect flags - diagonal projection
        proj_h = round(1.25 * h)
        step = 2 * is_up - 1
        projection_xy = np.zeros(proj_h)
        for i in range(1, w):
            y_end = stem_y + step * (proj_h + i)
            if stem_x + i >= img.shape[1] or y_end < 0 or y_end > img.shape[0]:
                break
            projection_xy = projection_xy + img[stem_y + step * i:y_end:step, stem_x + i]

        projection_xy = projection_xy / (w * 255)
        th = 0.4
        if np.max(projection_xy) > th:
            note.duration = 8

        # Detect beams
        if stem_x > 0:
            stem_end_roi_left = img[stem_y:stem_y + step * h:step, max(stem_x - w, 0):stem_x]
        else:
            stem_end_roi_left = [0]

        if stem_x + 1 < img.shape[1]:
            stem_end_roi_right = img[stem_y:stem_y + step * h:step, stem_x + 1:min(stem_x + w + 1, img.shape[1])]
        else:
            stem_end_roi_right = [0]

        stem_end_avg_left = np.average(stem_end_roi_left)/255
        stem_end_avg_right = np.average(stem_end_roi_right)/255

        th = 0.3
        if stem_end_avg_left > th or stem_end_avg_right > th:
            note.duration = 8

    strange=[]
    # Detect accidentals
    for note in detected_notes:
        center = note.center
        height = 0.75 * staff_height
        width = int(0.4 * staff_height)
        if note.bbox[0][0] - width < 0:
            continue
        roi = img[max(int(center[1] - height/2), 0):int(center[1] + height/2),
                  note.bbox[0][0] - width:note.bbox[0][0]]
        vert_proj = cv.reduce(roi, 0, cv.REDUCE_AVG)
        hor_proj = cv.reduce(roi, 1, cv.REDUCE_AVG)
        vert_mx = np.max(vert_proj)/255
        hor_mx = np.max(hor_proj)/255
        avg = np.average(roi)/255
        if vert_mx > 0.6 and avg > 0.2 and hor_mx > 0.4:
            is_note = False
            for other_note in detected_notes:
                if (other_note.staff == note.staff
                        and other_note.bbox[0][0] <= note.bbox[0][0] - width
                        and note.bbox[0][0] - width < (other_note.bbox[1][0] + other_note.center[0])/2):
                    is_note = True
            if not is_note:
                img_color = cv.rectangle(img_color, (note.bbox[0][0] - width, int(center[1] - height/2)),
                                         (note.bbox[0][0], int(center[1] + height/2)), (0, 200, 255), 2)
                strange.append(note)

    superarray=[]
    # Draw bounding box
    for note in detected_notes:
        bbox = note.stem_bbox
        thickness = 2
        if note.duration == 1:
            img_color = cv.rectangle(img_color, bbox[0], bbox[1], (255, 0, 0), thickness)
        elif note.duration == 2:
            img_color = cv.rectangle(img_color, bbox[0], bbox[1], (255, 255, 0), thickness)
        elif note.duration == 4:
            img_color = cv.rectangle(img_color, bbox[0], bbox[1], (0, 255, 0), thickness)
        elif note.duration == 8:
            img_color = cv.rectangle(img_color, bbox[0], bbox[1], (255, 0, 255), thickness)
        if note.duration > 0:
            img_color = cv.putText(img_color, str(note.pitch), (note.stem_bbox[0][0], note.stem_bbox[0][1] - 5),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            superarray.append(note)

    #return img_color
    return superarray, strange, clefs, img_color


def parser(f):
    preprocessed_data = []
    i = 0
    for d in f:
        #decorate(i)
        i = i + 1
        preprocessed_data.append(remove_staff_lines(d))

    i = 0
    for img, img_original, staff_lines in preprocessed_data:
        #decorate(i)
        i = i + 1
        sar, sar2, clf, img_color = detect_notes(img, img_original, staff_lines)

        for x in sar2:
            x.duration='#'
        sar.extend(sar2)

        sar=sorted(sar, key=(lambda x:(x.staff*10000000+10000*x.center[0]+x.center[1])))
        for i, x in enumerate(sar):
            if (i==0 or sar[i-1].staff!=sar[i].staff):
                print()
                zvx=print('TR', end=' ') if (clf[sar[i].staff][0]==1) else print('CL', end=' ')
            print(x.pitch, x.duration, end=' ', sep='')

        #plt.figure(figsize = (22, 22))
        #plot_color(img_color[:, :, (2, 1, 0)])
        #plt.show()

parser([cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)])
print()
