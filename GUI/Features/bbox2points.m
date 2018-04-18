function bboxPoints = bbox2points(bbox)

%%%bbox is [x, y, w, h]
x=bbox(1,1);
y=bbox(1,2);
w=bbox(1,3);
h=bbox(1,4);
bboxPoints=[x y; x y+w;x+h y+w; x+h y ];