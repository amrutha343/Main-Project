

clc
close all;
clear all;

XPV=[];YV=[];

kdir=uigetdir;
k2=strcat(kdir,'\*.tiff');
srcFiles = dir(k2);  % the folder in which ur images exists
for j = 1 : length(srcFiles)
    
    try
    fprintf('Processing %d of %d\n',j,length(srcFiles));
    filename = strcat(strcat(kdir,'\'),srcFiles(j).name);
    I = imread(filename);
    
    nam=srcFiles(j).name;
    clasnam=nam(4:5);
    
    
    
    switch(clasnam)
        case 'AN'
            yv=1;
        case 'DI'
            yv=2;
        case 'FE'
            yv=3;
        case 'HA'
            yv=4;
        case 'NE'
            yv=5;
        case 'SA'
            yv=6;
        case 'SU'
            yv=7;
    end
          
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Create the face detector object.
    faceDetector = vision.CascadeObjectDetector();
    
%     [filename, filepath]=uigetfile('*.bmp;*.jpg;*.tiff','Load image');
%     if filename==0
%         continue;
%     end
%     
%     I=imread([filepath,filename]);
    if size(I,3)==3 %RGB image
        I=rgb2gray(I);
    end
    
    % I = imread('cameraman.tif');
%     figure, imshow(I);title('I');
    
    H = fspecial('gaussian', [3 3], .5);
    I = imfilter(I,H,'replicate');
    
    [r,c]=size(I);
    
    %% Face Detection
    bboxf = faceDetector.step(I);
    
    if isempty(bboxf)
        disp('no face in the frame')
        continue
    end
    
    bboxf=bboxf(1,:);
    k=imcrop(I,bboxf);
    k=imresize(k,[96,96]);
    k=histeq(k);
    
    figure(1),imshow(k);title('Viola-Jones face');
    [h1,w1]=size(k);
    face1=k; %For future use.
    pause(.1);
    
    %Intial right eye crop
    x=round(w1/8);y=round( h1/6);w= round(w1/2.5);h= round(h1/2.5);
    Ebox1=[x y w h]; %Ebox
    k1=imcrop(k,Ebox1);
    figure(2),imshow(k1); title('initial Right eye crop');
    pause(.1);
    
    % Haar RightEye detector.
    faceDetector = vision.CascadeObjectDetector('RightEye');
    bboxe1 = faceDetector.step(k1);
    if isempty(bboxe1)
        disp('no RightEye in the frame')
        continue
    end
    bboxe1 = bboxe1(1,:);
    Eye1x=x+bboxe1(1)+floor(bboxe1(3)/2);
    Eye1y=y+bboxe1(2)+floor(bboxe1(4)/2.3);
    k2=imcrop(k1,bboxe1);
    figure(3),imshow(k2); title('Haar RightEye');
    pause(.1);
    figure(1);
    hold on
    plot(Eye1x,Eye1y,'*r');
    hold off
    pause(.1);
    
    %Intial left eye crop
    x=round(w1/2);y=round( h1/6);w= round(w1/2.5);h= round(h1/2.5);
    Ebox2=[x y w h]; %Ebox
    k1=imcrop(k,Ebox2);
    figure(4),imshow(k1); title('initial left eye crop');
    pause(.1);
    
    % Haar LeftEye detector.
    faceDetector = vision.CascadeObjectDetector('LeftEye');
    bboxe2= faceDetector.step(k1);
    if isempty(bboxe2)
        disp('no LeftEye in the frame')
        continue
    end
    bboxe2=bboxe2(1,:);
    Eye2x=x+bboxe2(1)+round(bboxe2(3)/2.3);
    Eye2y=y+bboxe2(2)+round(bboxe2(4)/2.3);
    k2=imcrop(k1,bboxe2);
    figure(5),imshow(k2); title('Haar LeftEye');
    pause(.1);
    figure(1);
    hold on
    plot(Eye2x,Eye2y,'*r');
    hold off
    pause(.1);
    
    %Intial nose crop
    x=round(w1/3.5);y=round( h1/2.5);w= round(w1/2.5);h= round(h1/2.5);
    Ebox2=[x y w h];
    k1=imcrop(k,Ebox2);
    figure(6),imshow(k1); title('initial nose crop');
    pause(.1);
    
    % Haar Nose detector.
    faceDetector = vision.CascadeObjectDetector('Nose');
    bboxn= faceDetector.step(k1);
    if isempty(bboxn)
        disp('no Nose in the frame')
        continue
    end
    bboxn=bboxn(1,:);
    nsx=x+bboxn(1)+round(bboxn(3)/2);
    nsy=y+bboxn(2)+round(bboxn(4)/2);
    k2=imcrop(k1,bboxn);
    figure(7),imshow(k2); title('Haar Nose');
    pause(.1);
    figure(1);
    hold on
    plot(nsx,nsy,'*r');
    hold off
    pause(.1);
    
    %% 4.3 Lip Corner Detection
    
    %1: select coarse lips ROI using face width and nose position
    [h1,w1]=size(k);
    x=round(w1/4.5);y=nsy+round(h1/16);w= w1-2*x;h= h1-y;
    lpx=x;
    lpy=y;
    Ebox2=[x y w h];
    k1=imcrop(k,Ebox2);
    figure(8),imshow(k1); title('initial lips crop');
    pause(.1);
    
    %2: apply Gaussian blur to the lips ROI
    G = fspecial('gaussian', [5 5], 2);
    % Blur Image
    bd = imfilter(k1,G,'same');
%     figure(8),imshow(bd); title('Blurred lips');
%     pause(.1);
    
    %3: apply horizontal sobel operator for edge detection
    h = fspecial('sobel') ;
    sob = imfilter(bd,h,'replicate');
%     figure(9),imshow(sob); title('Edge lips');
%     pause(.1);
    
    %4: apply Otsu-thresholding
    bw=im2bw(sob);
    figure(10),imshow(bw); title('Otsu-Thresholding');
    pause(.1);
    
    %5: apply morphological dilation operation
    [sr,sc]=size(bw);
    sr1=floor(sr/15);
    sc1=floor(sc/12);
    se=strel('rectangle',[sr1,sc1]);
    dil=imclose(bw,se);
%     figure(11),imshow(dil); title('Dilated');
%     pause(.1);
    
    %7: remove the spurious connected components using threshold technique to the number of pixels
    maxArea=ceil(sr*sc/50);
    bw1= bwareaopen(dil,maxArea);
    figure(12),imshow(bw1);
    
    %6: find the connected components
    stats = regionprops(bw1,'Area','BoundingBox','Centroid','PixelIdxList');
    if isempty(stats)
        continue
    end
    
    
    %8: scan the image from the top and select the first connected component as upper lip position
    cn=[];
    for i=1:length(stats)
        cn(i,:)=stats(i).Centroid;
    end
    figure(10)
    hold on
    plot(cn(:,1),cn(:,2),'g*');
    
    yn=cn(:,2);
    ys=min(yn);
    yi=find(yn==ys);
    yi=yi(1);
    cnt=cn(yi,:);
    plot(cnt(1),cnt(2),'r*');
    hold off
    
    %B=stats(yi).BoundingBox;
    
    %9: locate the left and right most positions of connected component as lip corners
    [R,C]=size(bw);
    pp=stats(yi).PixelIdxList;
    [y3,x3]=ind2sub([R C],pp);
    
    figure(10)
    hold on
    plot(x3,y3,'y*');
    hold off
    
    xa=min(x3);
    xa=xa(1);
    ya=y3(x3==xa);
    ya=ya(1);
    xb=max(x3);
    xb=xb(1);
    yb=y3(x3==xb);
    yb=yb(1);
    X=[xa,xb];
    Y=[ya,yb];
    figure(10)
    hold on
    plot(X',Y','y*');
    hold off
    
    lpxa=lpx+xa;
    lpya=lpy+ya;
    lpxb=lpx+xb;
    lpyb=lpy+yb;
    %         X=[lpxa+sc1,lpxb-sc1];
    %         Y=[lpya+sr1,lpyb+sr1];
    LipX=[lpxa,lpxb-sc1];
    LipY=[lpya+sr1,lpyb+sr1];
    
    %         figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    plot(LipX',LipY','*b');
    hold off
    pause(.1);
    
    
    %% 4.4 Eyebrow Corner Detection
    
    %Right Eyebrow
    %1: select coarse  Right Eyebrow ROI using face width and eye position
    [h1,w1]=size(k);
    x=Eye1x;y=Eye1y-round(h1/4);w=w1/2-x;h=Eye1y-y-round(h1/30);
    Eb1x=x;
    Eb1y=y;
    Ebbox1=[x y w h];
    k1=imcrop(k,Ebbox1);
    figure(13),imshow(k1); title('initial Eyebrow crop');
    pause(.1);
    
    %         bw1=adaptivethreshold(k1,round(w/5),0.01,0);
    
    %2: apply Gaussian blur to the lips ROI
    G = fspecial('gaussian', [3 3], 2);
    % Blur Image
    bd = imfilter(k1,G,'same');
%     figure(8),imshow(bd); title('Blurred lips');
%     pause(.1);
    %
    %3: apply horizontal sobel operator for edge detection
    h = fspecial('sobel') ;
    sob = imfilter(bd,h,'replicate');
%     figure(9),imshow(sob); title('Edge');
%     pause(.1);
    
    
    bw1=im2bw(sob);
    figure(13),imshow(bw1); title('bw Eyebrow ');
    pause(.1);
    
    [h,w]=size(bw1);
    bw1=bwareaopen(bw1,round(h*w/100));
    b = padarray(bw1,[0 1],'pre');
    b2=imclearborder(b);
    figure(14),imshow(b2); title('bw Eyebrow ');
    pause(.1);
    
    
    ind=find(b2==1);
    [Y,X]=ind2sub(size(bw1),ind);
    x=max(X);
    y=Y(X==x);
    y=y(1);
    
    figure(14)
    hold on
    plot(x,y,'*b');
    hold off
    
    EB1x=Eb1x+x-round(w1/20);
    EB1y=Eb1y+y;
    
    %         figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    plot(EB1x,EB1y,'*b');
    hold off
    pause(.1)
    
    
    %Left Eyebrow
    %1: select coarse  Right Eyebrow ROI using face width and eye position
    [h1,w1]=size(k);
    x=Eye2x-round(w1/4);y=Eye2y-round(h1/4);w=round((w1-x-round(w1/15))/2);h=Eye2y-y-round(h1/30);
    Eb2x=x;
    Eb2y=y;
    Ebbox1=[x y w h];
    k1=imcrop(k,Ebbox1);
    figure(13),imshow(k1); title('initial Eyebrow crop');
    pause(.1);
    
    %2: apply Gaussian blur to the lips ROI
    G = fspecial('gaussian', [3 3], 2);
    % Blur Image
    bd = imfilter(k1,G,'same');
%     figure(8),imshow(bd); title('Blurred lips');
%     pause(.1);
    %
    %3: apply horizontal sobel operator for edge detection
    h = fspecial('sobel') ;
    sob = imfilter(bd,h,'replicate');
%     figure(9),imshow(sob); title('Edge');
%     pause(.1);
    
    
    bw1=im2bw(sob);
    figure(13),imshow(bw1); title('bw Eyebrow ');
    pause(.1);
    
    [h,w]=size(bw1);
    bw1=bwareaopen(bw1,round(h*w/100));
    b = padarray(bw1,[0 1],'post');
    b2=imclearborder(b);
    figure(14),imshow(b2); title('bw Eyebrow ');
    pause(.1);
    
    
    ind=find(b2==1);
    [Y,X]=ind2sub(size(bw1),ind);
    x=min(X);
    y=Y(X==x);
    y=y(1);
    
    figure(14)
    hold on
    plot(x,y,'*b');
    hold off
    pause(.1);
    
    EB2x=Eb2x+x;
    EB2y=Eb2y+y;
    
    %         figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    plot(EB2x,EB2y,'*b');
    hold off
    pause(.1)
    
    
    %% 4.5 Extraction of Active Facial Patches
    [h1,w1]=size(k);
    wd=round(w1/9);
    if rem(wd,2)==0
        wd=wd+1;
    end
    wg=floor(wd/2);
    
    %P18
%     figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[EB1x-wg,EB1y-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P18=imcrop(k,[EB1x-wg,EB1y-wg,wd,wd]);
    %         figure(18),imshow(P18); title('P18 ');
    %         pause(.3);
    
    %P19
    figure(1);
    hold on
    rectangle('position',[EB2x-wg,EB2y-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P19=imcrop(k,[EB2x-wg,EB2y-wg,wd,wd]);
    %         figure(19),imshow(P19); title('P19 ');
    %         pause(.3);
    
    
    %P1
    figure(1);
    hold on
    rectangle('position',[LipX(1)-wg,LipY(1)-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P1=imcrop(k,[LipX(1)-wg,LipY(1)-wg,wd,wd]);
    %         figure(16),imshow(P1); title('P1 ');
    %         pause(.3);
    
    %P9
    figure(1);
    hold on
    rectangle('position',[LipX(1)-wg,LipY(1)-wg+wd,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P9=imcrop(k,[LipX(1)-wg,LipY(1)-wg+wd,wd,wd]);
    %         figure(16),imshow(P1); title('P1 ');
    %         pause(.3);
    
    %P4
    figure(1);
    hold on
    rectangle('position',[LipX(2)-wg,LipY(2)-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P4=imcrop(k,[LipX(2)-wg,LipY(2)-wg,wd,wd]);
    %         figure(16),imshow(P4); title('P4 ');
    %         pause(.3);
    
    
    %P11
    figure(1);
    hold on
    rectangle('position',[LipX(2)-wg,LipY(2)-wg+wd,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P11=imcrop(k,[LipX(2)-wg,LipY(2)-wg+wd,wd,wd]);
    %         figure(16),imshow(P11); title('P11 ');
    %         pause(.3);
    
    
    %P10
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[round((LipX(2)+LipX(1))/2)-wg,round((LipY(2)+LipY(1))/2)+wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P10=imcrop(k,[round((LipX(2)+LipX(1))/2)-wg,round((LipY(2)+LipY(1))/2)+wg,wd,wd]);
    %         figure(16),imshow(P11); title('P11 ');
    %         pause(.3);
    
    
    
    %P16
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[round(Eye2x+Eye1x)/2-wg,round(round(Eye2y+Eye1y)/2-2*wg),wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P16=imcrop(k,[round(Eye2x+Eye1x)/2-wg,round(round(Eye2y+Eye1y)/2-2*wg),wd,wd]);
    %         figure(17),imshow(P16); title('P16 ');
    %         pause(.3);
    
    
    %P17
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[round(Eye2x+Eye1x)/2-wg,round(round(Eye2y+Eye1y)/2-4*wg),wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P17=imcrop(k,[round(Eye2x+Eye1x)/2-wg,round(Eye2y+Eye1y)/2-2*wg,wd,wd]);
    %         figure(18),imshow(P17); title('P17 ');
    %         pause(.3);
    
    
    %P14
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[Eye1x-round(1.5*wg),(round(Eye1y+.3*wg)),wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P14=imcrop(k,[Eye1x-round(1.5*wg),(round(Eye1y+.3*wg)),wd,wd]);
    %         figure(19),imshow(P3); title('P3 ');
    %         pause(.3);
    
    
    %P15
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[round(Eye2x-0.8*wg),(round(Eye2y+.3*wg)),wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P15=imcrop(k,[round(Eye2x-0.8*wg),(round(Eye2y+.3*wg)),wd,wd]);
    %         figure(19),imshow(P3); title('P3 ');
    %         pause(.3);
    
    
    %P3
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[round(nsx+Eye1x)/2-wg,round(round(nsy+Eye1y)/2-2*wg),wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P3=imcrop(k,[round(nsx+Eye1x)/2-wg,round(nsy+Eye1y)/2-2*wg,wd,wd]);
    %         figure(19),imshow(P3); title('P3 ');
    %         pause(.3);
    
    
    %P6
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[round(nsx+Eye2x)/2-wg,round(round(nsy+Eye2y)/2-2*wg),wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P6=imcrop(k,[round(nsx+Eye2x)/2-wg,round(round(nsy+Eye2y)/2-2*wg),wd,wd]);
    %         figure(19),imshow(P6); title('P6 ');
    %         pause(.3);
    
    
    
    %P13
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[nsx+2*wg+wd,nsy-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P13=imcrop(k,[nsx+2*wg+wd,nsy-wg,wd,wd]);
    %         figure(19),imshow(P16); title('P16 ');
    %         pause(.3);
    
    
    %P12
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[nsx+2*wg+wd,nsy-wg+wd,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P12=imcrop(k,[nsx+2*wg+wd,nsy-wg+wd,wd,wd]);
    %         figure(19),imshow(P16); title('P16 ');
    %         pause(.3);
    
    
    %P7
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[nsx-4*wg-wd,nsy-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P7=imcrop(k,[nsx-4*wg-wd,nsy-wg,wd,wd]);
    %         figure(19),imshow(P16); title('P16 ');
    %         pause(.3);
    
    
    %P8
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[nsx-4*wg-wd,nsy-wg+wd,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P8=imcrop(k,[nsx-4*wg-wd,nsy-wg+wd,wd,wd]);
    %         figure(19),imshow(P16); title('P16 ');
    %         pause(.3);
    
    
    %P5
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[nsx+2*wg,nsy-wg,wd,wd],'edgecolor','g');
%     pause(.1);
    hold off
    P5=imcrop(k,[nsx+2*wg,nsy-wg,wd,wd]);
    %         figure(19),imshow(P5); title('P5 ');
    %         pause(.3);
    
    
    %P2
    %figure(1),imshow(k);title('Viola-Jones face');
    figure(1);
    hold on
    rectangle('position',[nsx-4*wg,nsy-wg,wd,wd],'edgecolor','g');
%     pause(.3);
    hold off
    P2=imcrop(k,[nsx-4*wg,nsy-wg,wd,wd]);
    %         figure(19),imshow(P2); title('P2 ');
    %         pause(.3);
    
    
    %% Feature extraction from all patches
    %LBP
    R=2;
    
    LP1=LBP(P1,R);
    HP1=hist((LP1(:)),16);
    figure(20);bar(HP1);
    
    LP2=LBP(P2,R);
    HP2=hist((LP2(:)),16);
    figure(20);bar(HP2);
    
    LP3=LBP(P3,R);
    HP3=hist((LP3(:)),16);
    figure(20);bar(HP3);
    
    LP4=LBP(P4,R);
    HP4=hist((LP4(:)),16);
    figure(20);bar(HP4);
    
    LP5=LBP(P5,R);
    HP5=hist((LP5(:)),16);
    figure(20);bar(HP5);
    
    LP6=LBP(P6,R);
    HP6=hist((LP6(:)),16);
    figure(20);bar(HP6);
    
    LP7=LBP(P7,R);
    HP7=hist((LP7(:)),16);
    figure(20);bar(HP7);
    
    LP8=LBP(P8,R);
    HP8=hist((LP8(:)),16);
    figure(20);bar(HP8);
    
    LP9=LBP(P9,R);
    HP9=hist((LP9(:)),16);
    figure(20);bar(HP9);
    
    LP10=LBP(P10,R);
    HP10=hist((LP10(:)),16);
    figure(20);bar(HP10);
    
    LP11=LBP(P11,R);
    HP11=hist((LP11(:)),16);
    figure(20);bar(HP11);
    
    LP12=LBP(P12,R);
    HP12=hist((LP12(:)),16);
    figure(20);bar(HP12);
    
    LP13=LBP(P13,R);
    HP13=hist((LP13(:)),16);
    figure(20);bar(HP13);
    
    LP14=LBP(P14,R);
    HP14=hist((LP14(:)),16);
    figure(20);bar(HP14);
    
    LP15=LBP(P15,R);
    HP15=hist((LP15(:)),16);
    figure(20);bar(HP15);
    
    LP16=LBP(P16,R);
    HP16=hist((LP16(:)),16);
    figure(20);bar(HP16);
    
    LP17=LBP(P17,R);
    HP17=hist((LP17(:)),16);
    figure(20);bar(HP17);
    
    LP18=LBP(P18,R);
    HP18=hist((LP18(:)),16);
    figure(20);bar(HP18);
    
    LP19=LBP(P19,R);
    HP19=hist((LP19(:)),16);
    figure(20);bar(HP19);
    
    %%
    %Rotation invariant LBP_u2
    
    LPr1=LBP_u2(P1,R);
    HPr1=hist((LPr1(:)),10);
    figure(20);bar(HP1);
    
    LPr2=LBP_u2(P2,R);
    HPr2=hist((LPr2(:)),10);
    figure(20);bar(HPr2);
    
    LPr3=LBP_u2(P3,R);
    HPr3=hist((LPr3(:)),10);
    figure(20);bar(HPr3);
    
    LPr4=LBP_u2(P4,R);
    HPr4=hist((LPr4(:)),10);
    figure(20);bar(HPr4);
    
    LPr5=LBP_u2(P5,R);
    HPr5=hist((LPr5(:)),10);
    figure(20);bar(HPr5);
    
    LPr6=LBP_u2(P6,R);
    HPr6=hist((LPr6(:)),10);
    figure(20);bar(HPr6);
    
    LPr7=LBP_u2(P7,R);
    HPr7=hist((LPr7(:)),10);
    figure(20);bar(HPr7);
    
    LPr8=LBP_u2(P8,R);
    HPr8=hist((LPr8(:)),10);
    figure(20);bar(HPr8);
    
    LPr9=LBP_u2(P9,R);
    HPr9=hist((LPr9(:)),10);
    figure(20);bar(HPr9);
    
    LPr10=LBP_u2(P10,R);
    HPr10=hist((LPr10(:)),10);
    figure(20);bar(HPr10);
    
    LPr11=LBP_u2(P11,R);
    HPr11=hist((LPr11(:)),10);
    figure(20);bar(HPr11);
    
    LPr12=LBP_u2(P12,R);
    HPr12=hist((LPr12(:)),10);
    figure(20);bar(HPr12);
    
    LPr13=LBP_u2(P13,R);
    HPr13=hist((LPr13(:)),10);
    figure(20);bar(HPr13);
    
    LPr14=LBP_u2(P14,R);
    HPr14=hist((LPr14(:)),10);
    figure(20);bar(HPr14);
    
    LPr15=LBP_u2(P15,R);
    HPr15=hist((LPr15(:)),10);
    figure(20);bar(HPr15);
    
    LPr16=LBP_u2(P16,R);
    HPr16=hist((LPr16(:)),10);
    figure(20);bar(HPr16);
    
    LPr17=LBP_u2(P17,R);
    HPr17=hist((LPr17(:)),10);
    figure(20);bar(HPr17);
    
    LPr18=LBP_u2(P18,R);
    HPr18=hist((LPr18(:)),10);
    figure(20);bar(HPr18);
    
    LPr19=LBP_u2(P19,R);
    HPr19=hist((LPr19(:)),10);
    figure(20);bar(HPr19);
    
    %% Features for each patch
    FP1=[HP1,HPr1];
    FP2=[HP2,HPr2];
    FP3=[HP3,HPr3];
    FP4=[HP4,HPr4];
    FP5=[HP5,HPr5];
    FP6=[HP6,HPr6];
    FP7=[HP7,HPr7];
    FP8=[HP8,HPr8];
    FP9=[HP9,HPr9];
    FP10=[HP10,HPr10];
    FP11=[HP11,HPr11];
    FP12=[HP12,HPr12];
    FP13=[HP13,HPr13];
    FP14=[HP14,HPr14];
    FP15=[HP15,HPr15];
    FP16=[HP16,HPr16];
    FP17=[HP17,HPr17];
    FP18=[HP18,HPr18];
    FP19=[HP19,HPr19];
    
    figure(1);
    pause(.3)
    catch
        warning('Skipping the face for something is missing');
        continue
    end
    
%     in=input('Enter s to skip the face, else enter any key','s');
%     if strcmp(in,'s')
%         continue;
%     end
    
    
    XP=[FP1,FP2,FP3,FP4,FP5,FP6,FP7,FP8,FP9,FP10,FP11,FP12,FP13,FP14,FP15,FP16,FP17,FP18,FP19];
    XPV=[XPV;XP];
    YV=[YV;yv];
    
    
end

save XPV XPV YV




