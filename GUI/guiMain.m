function varargout = guiMain(varargin)
% GUIMAIN MATLAB code for guiMain.fig
%      GUIMAIN, by itself, creates a new GUIMAIN or raises the existing
%      singleton*.
%
%      H = GUIMAIN returns the handle to a new GUIMAIN or the handle to
%      the existing singleton*.
%
%      GUIMAIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUIMAIN.M with the given input arguments.
%
%      GUIMAIN('Property','Value',...) creates a new GUIMAIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before guiMain_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to guiMain_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help guiMain

% Last Modified by GUIDE v2.5 21-Mar-2018 09:33:36

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @guiMain_OpeningFcn, ...
                   'gui_OutputFcn',  @guiMain_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before guiMain is made visible.
function guiMain_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to guiMain (see VARARGIN)

% Choose default command line output for guiMain
handles.output = hObject;
set(handles.axes5,'visible','off') 
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes guiMain wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = guiMain_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
warning('off','all');
wrongIcon = imread('Wrong.png');
checkValue1 = get(handles.checkbox1,'value');
checkValue2 = get(handles.checkbox2,'value');

if (checkValue1 == 1 && checkValue2 == 1)
    msgBox = msgbox({'Invalid Selection' 'You can select correct option at a time'},'ERROR','custom',wrongIcon);
else if (checkValue1 == 0 && checkValue2 == 0)
        msgBox = msgbox({'Invalid Selection' 'Please select any option'},'ERROR','custom',wrongIcon);
    else if (checkValue1 == 1 && checkValue2 == 0)
            imshow('','parent',handles.axes3);
            [fn pn] = uigetfile('*.jpg;*.png;*.bmp','Select any image');
            I = imread([pn fn]);
            imshow(I,'parent',handles.axes3);
            addpath('Features');
            load('SVMModel.mat');
            faceDetector = vision.CascadeObjectDetector();
            try
            if size(I,3)==3 
                I=rgb2gray(I);
            end
            
            H = fspecial('gaussian', [3 3], .5);
            I = imfilter(I,H,'replicate');
            
            [r,c]=size(I);
            bboxf = faceDetector.step(I);
            
            if isempty(bboxf)
                disp('no face in the frame')
                msgBox = msgbox({'no face in the frame'},'ERROR','custom',wrongIcon);
                return;
            end
            
            bboxf=bboxf(1,:);
            k=imcrop(I,bboxf);
            k=imresize(k,[96,96]);
            k=histeq(k);
            
            [h1,w1]=size(k);
            face1=k;
            
            %Intial right eye crop
            x=round(w1/8);y=round( h1/6);w= round(w1/2.5);h= round(h1/2.5);
            Ebox1=[x y w h]; 
            k1=imcrop(k,Ebox1);
            
            % Haar RightEye detector.
            faceDetector = vision.CascadeObjectDetector('RightEye');
            bboxe1 = faceDetector.step(k1);
            if isempty(bboxe1)
                disp('no RightEye in the frame')
                msgBox = msgbox({'no right eye in the frame'},'ERROR','custom',wrongIcon);
                return
            end
            bboxe1 = bboxe1(1,:);
            Eye1x=x+bboxe1(1)+floor(bboxe1(3)/2);
            Eye1y=y+bboxe1(2)+floor(bboxe1(4)/2.3);
            filePath=imcrop(k1,bboxe1);
            
            %Intial left eye crop
            x=round(w1/2);y=round( h1/6);w= round(w1/2.5);h= round(h1/2.5);
            Ebox2=[x y w h]; %Ebox
            k1=imcrop(k,Ebox2);
           
            % Haar LeftEye detector.
            faceDetector = vision.CascadeObjectDetector('LeftEye');
            bboxe2= faceDetector.step(k1);
            if isempty(bboxe2)
                disp('no LeftEye in the frame')
                msgBox = msgbox({'no left eye in the frame'},'ERROR','custom',wrongIcon);
                return
            end
            bboxe2=bboxe2(1,:);
            Eye2x=x+bboxe2(1)+round(bboxe2(3)/2.3);
            Eye2y=y+bboxe2(2)+round(bboxe2(4)/2.3);
            filePath=imcrop(k1,bboxe2);
            
            %Intial nose crop
            x=round(w1/3.5);y=round( h1/2.5);w= round(w1/2.5);h= round(h1/2.5);
            Ebox2=[x y w h];
            k1=imcrop(k,Ebox2);
            
            % Haar Nose detector.
            faceDetector = vision.CascadeObjectDetector('Nose');
            bboxn= faceDetector.step(k1);
            if isempty(bboxn)
                disp('no Nose in the frame')
                msgBox = msgbox({'no nose in the frame'},'ERROR','custom',wrongIcon);
                return
            end
            bboxn=bboxn(1,:);
            nsx=x+bboxn(1)+round(bboxn(3)/2);
            nsy=y+bboxn(2)+round(bboxn(4)/2);
            filePath=imcrop(k1,bboxn);
            
            % 4.3 Lip Corner Detection
            
            %1: select coarse lips ROI using face width and nose position
            [h1,w1]=size(k);
            x=round(w1/4.5);y=nsy+round(h1/16);w= w1-2*x;h= h1-y;
            lpx=x;
            lpy=y;
            Ebox2=[x y w h];
            k1=imcrop(k,Ebox2);
            
            %2: apply Gaussian blur to the lips ROI
            G = fspecial('gaussian', [5 5], 2);
            bd = imfilter(k1,G,'same');
            
            %3: apply horizontal sobel operator for edge detection
            h = fspecial('sobel') ;
            sob = imfilter(bd,h,'replicate');
            
            %4: apply Otsu-thresholding
            bw=im2bw(sob);
            
            %5: apply morphological dilation operation
            [sr,sc]=size(bw);
            sr1=floor(sr/15);
            sc1=floor(sc/12);
            se=strel('rectangle',[sr1,sc1]);
            dil=imclose(bw,se);
            
            %7: remove the spurious connected components using threshold technique to the number of pixels
            maxArea=ceil(sr*sc/50);
            bw1= bwareaopen(dil,maxArea);
            
            %6: find the connected components
            stats = regionprops(bw1,'Area','BoundingBox','Centroid','PixelIdxList');
            if isempty(stats)
                return
            end

            %8: scan the image from the top and select the first connected component as upper lip position
            cn=[];
            for i=1:length(stats)
                cn(i,:)=stats(i).Centroid;
            end
            yn=cn(:,2);
            ys=min(yn);
            yi=find(yn==ys);
            yi=yi(1);
            cnt=cn(yi,:);
            
            %9: locate the left and right most positions of connected component as lip corners
            [R,C]=size(bw);
            pp=stats(yi).PixelIdxList;
            [y3,x3]=ind2sub([R C],pp);
            
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
           
            lpxa=lpx+xa;
            lpya=lpy+ya;
            lpxb=lpx+xb;
            lpyb=lpy+yb;
            
            LipX=[lpxa,lpxb-sc1];
            LipY=[lpya+sr1,lpyb+sr1];
            
            % 4.4 Eyebrow Corner Detection
            
            %Right Eyebrow
            %1: select coarse  Right Eyebrow ROI using face width and eye position
            [h1,w1]=size(k);
            x=Eye1x;y=Eye1y-round(h1/4);w=w1/2-x;h=Eye1y-y-round(h1/30);
            Eb1x=x;
            Eb1y=y;
            Ebbox1=[x y w h];
            k1=imcrop(k,Ebbox1);
            
            %2: apply Gaussian blur to the lips ROI
            G = fspecial('gaussian', [3 3], 2);
            bd = imfilter(k1,G,'same');
            
            %3: apply horizontal sobel operator for edge detection
            h = fspecial('sobel') ;
            sob = imfilter(bd,h,'replicate');
            
            bw1=im2bw(sob);
            
            [h,w]=size(bw1);
            bw1=bwareaopen(bw1,round(h*w/100));
            b = padarray(bw1,[0 1],'pre');
            b2=imclearborder(b);
           
            ind=find(b2==1);
            [Y,X]=ind2sub(size(bw1),ind);
            x=max(X);
            y=Y(X==x);
            y=y(1);
            
            EB1x=Eb1x+x-round(w1/20);
            EB1y=Eb1y+y;
            
            %Left Eyebrow
            %1: select coarse  Right Eyebrow ROI using face width and eye position
            [h1,w1]=size(k);
            x=Eye2x-round(w1/4);y=Eye2y-round(h1/4);w=round((w1-x-round(w1/15))/2);h=Eye2y-y-round(h1/30);
            Eb2x=x;
            Eb2y=y;
            Ebbox1=[x y w h];
            k1=imcrop(k,Ebbox1);
            
            %2: apply Gaussian blur to the lips ROI
            G = fspecial('gaussian', [3 3], 2);
            bd = imfilter(k1,G,'same');
           
            %3: apply horizontal sobel operator for edge detection
            h = fspecial('sobel') ;
            sob = imfilter(bd,h,'replicate');
            bw1=im2bw(sob);
            [h,w]=size(bw1);
            bw1=bwareaopen(bw1,round(h*w/100));
            b = padarray(bw1,[0 1],'post');
            b2=imclearborder(b);
            
            ind=find(b2==1);
            [Y,X]=ind2sub(size(bw1),ind);
            x=min(X);
            y=Y(X==x);
            y=y(1);
                        
            EB2x=Eb2x+x;
            EB2y=Eb2y+y;
            
            % 4.5 Extraction of Active Facial Patches
            [h1,w1]=size(k);
            wd=round(w1/9);
            if rem(wd,2)==0
                wd=wd+1;
            end
            wg=floor(wd/2);
            
            P18=imcrop(k,[EB1x-wg,EB1y-wg,wd,wd]);
            P19=imcrop(k,[EB2x-wg,EB2y-wg,wd,wd]);
            P1=imcrop(k,[LipX(1)-wg,LipY(1)-wg,wd,wd]);
            P9=imcrop(k,[LipX(1)-wg,LipY(1)-wg+wd,wd,wd]);
            P4=imcrop(k,[LipX(2)-wg,LipY(2)-wg,wd,wd]);
            P11=imcrop(k,[LipX(2)-wg,LipY(2)-wg+wd,wd,wd]);
            P10=imcrop(k,[round((LipX(2)+LipX(1))/2)-wg,round((LipY(2)+LipY(1))/2)+wg,wd,wd]);
            P16=imcrop(k,[round(Eye2x+Eye1x)/2-wg,round(round(Eye2y+Eye1y)/2-2*wg),wd,wd]);
            P17=imcrop(k,[round(Eye2x+Eye1x)/2-wg,round(Eye2y+Eye1y)/2-2*wg,wd,wd]);
            P14=imcrop(k,[Eye1x-round(1.5*wg),(round(Eye1y+.3*wg)),wd,wd]);
            P15=imcrop(k,[round(Eye2x-0.8*wg),(round(Eye2y+.3*wg)),wd,wd]);
            P3=imcrop(k,[round(nsx+Eye1x)/2-wg,round(nsy+Eye1y)/2-2*wg,wd,wd]);
            P6=imcrop(k,[round(nsx+Eye2x)/2-wg,round(round(nsy+Eye2y)/2-2*wg),wd,wd]);
            P13=imcrop(k,[nsx+2*wg+wd,nsy-wg,wd,wd]);
            P12=imcrop(k,[nsx+2*wg+wd,nsy-wg+wd,wd,wd]);
            P7=imcrop(k,[nsx-4*wg-wd,nsy-wg,wd,wd]);
            P8=imcrop(k,[nsx-4*wg-wd,nsy-wg+wd,wd,wd]);
            P5=imcrop(k,[nsx+2*wg,nsy-wg,wd,wd]);
            P2=imcrop(k,[nsx-4*wg,nsy-wg,wd,wd]);
            
            % Feature extraction from all patches
            %LBP
            R=2;
            
            LP1=LBP(P1,R);
            HP1=hist((LP1(:)),16);
            
            LP2=LBP(P2,R);
            HP2=hist((LP2(:)),16);
            
            LP3=LBP(P3,R);
            HP3=hist((LP3(:)),16);
            
            LP4=LBP(P4,R);
            HP4=hist((LP4(:)),16);
            
            LP5=LBP(P5,R);
            HP5=hist((LP5(:)),16);
            
            LP6=LBP(P6,R);
            HP6=hist((LP6(:)),16);
            
            LP7=LBP(P7,R);
            HP7=hist((LP7(:)),16);
            
            LP8=LBP(P8,R);
            HP8=hist((LP8(:)),16);
            
            LP9=LBP(P9,R);
            HP9=hist((LP9(:)),16);
            
            LP10=LBP(P10,R);
            HP10=hist((LP10(:)),16);
            
            LP11=LBP(P11,R);
            HP11=hist((LP11(:)),16);
            
            LP12=LBP(P12,R);
            HP12=hist((LP12(:)),16);
            
            LP13=LBP(P13,R);
            HP13=hist((LP13(:)),16);
            
            LP14=LBP(P14,R);
            HP14=hist((LP14(:)),16);
            
            LP15=LBP(P15,R);
            HP15=hist((LP15(:)),16);
            
            LP16=LBP(P16,R);
            HP16=hist((LP16(:)),16);
            
            LP17=LBP(P17,R);
            HP17=hist((LP17(:)),16);
            
            LP18=LBP(P18,R);
            HP18=hist((LP18(:)),16);
            
            LP19=LBP(P19,R);
            HP19=hist((LP19(:)),16);
            
            
            %Rotation invariant LBP_u2
            LPr1=LBP_u2(P1,R);
            HPr1=hist((LPr1(:)),10);
            
            LPr2=LBP_u2(P2,R);
            HPr2=hist((LPr2(:)),10);
            
            LPr3=LBP_u2(P3,R);
            HPr3=hist((LPr3(:)),10);
            
            LPr4=LBP_u2(P4,R);
            HPr4=hist((LPr4(:)),10);
            
            LPr5=LBP_u2(P5,R);
            HPr5=hist((LPr5(:)),10);
            
            LPr6=LBP_u2(P6,R);
            HPr6=hist((LPr6(:)),10);
            
            LPr7=LBP_u2(P7,R);
            HPr7=hist((LPr7(:)),10);
            
            LPr8=LBP_u2(P8,R);
            HPr8=hist((LPr8(:)),10);
            
            LPr9=LBP_u2(P9,R);
            HPr9=hist((LPr9(:)),10);
            
            LPr10=LBP_u2(P10,R);
            HPr10=hist((LPr10(:)),10);
            
            LPr11=LBP_u2(P11,R);
            HPr11=hist((LPr11(:)),10);
            
            LPr12=LBP_u2(P12,R);
            HPr12=hist((LPr12(:)),10);
            
            LPr13=LBP_u2(P13,R);
            HPr13=hist((LPr13(:)),10);
            
            LPr14=LBP_u2(P14,R);
            HPr14=hist((LPr14(:)),10);
            
            LPr15=LBP_u2(P15,R);
            HPr15=hist((LPr15(:)),10);
            
            LPr16=LBP_u2(P16,R);
            HPr16=hist((LPr16(:)),10);
            
            LPr17=LBP_u2(P17,R);
            HPr17=hist((LPr17(:)),10);
            
            LPr18=LBP_u2(P18,R);
            HPr18=hist((LPr18(:)),10);
            
            LPr19=LBP_u2(P19,R);
            HPr19=hist((LPr19(:)),10);
            
            % Features for each patch
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
            
            
            fe = [FP1,FP2,FP3,FP4,FP5,FP6,FP7,FP8,FP9,FP10,FP11,FP12,FP13,FP14,FP15,FP16,FP17,FP18,FP19];
            catch
            end
            predict = svm.predict(ModelHappy,fe);
            musicIcon = imread('Music.png');
            if predict==0
                set(handles.axes5,'visible','on')
                imshow(musicIcon,'parent',handles.axes5);
                disp('Happy');
                winopen('Happy.mp3');
                set(handles.edit11,'string','Happy');
            else
                predict = svm.predict(ModelSad,fe);
                if predict==0
                    set(handles.axes5,'visible','on')
                    imshow(musicIcon,'parent',handles.axes5);
                    disp('Sad');
                    winopen('Sad.mp3');
                    set(handles.edit11,'string','Sad');
                else if predict==1
                        set(handles.axes5,'visible','on')
                        imshow(musicIcon,'parent',handles.axes5);
                        disp('Angry');set(handles.edit11,'string','Angry');
                        winopen('Angry.mp3');
                    end
                end
            end
        set(handles.axes5,'visible','off')   
        else
            msgBox = msgbox({'Invalid Selection' 'You can select correct option at a time'},'ERROR','custom',wrongIcon);
        end
    end
end


% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
warning('off','all');
wrongIcon = imread('Wrong.png');
checkValue1 = get(handles.checkbox1,'value');
checkValue2 = get(handles.checkbox2,'value');

if (checkValue1 == 1 && checkValue2 == 1)
    msgBox = msgbox({'Invalid Selection' 'You can select correct option at a time'},'ERROR','custom',wrongIcon);
else if (checkValue1 == 0 && checkValue2 == 0)
        msgBox = msgbox({'Invalid Selection' 'Please select any option'},'ERROR','custom',wrongIcon);
    else if (checkValue1 == 0 && checkValue2 == 1)
            imshow('','parent',handles.axes4);
            vid = videoinput('winvideo', 1, 'YUY2_640x480');
            vid.FramesPerTrigger = 1;
            vid.ReturnedColorspace = 'rgb';
            triggerconfig(vid, 'manual');
            vidRes = get(vid, 'VideoResolution');
            imWidth = vidRes(1);
            imHeight = vidRes(2);
            nBands = get(vid, 'NumberOfBands');
            hImage = image(zeros(imHeight, imWidth, nBands), 'parent', handles.axes4);
            numFrames = get(handles.edit13,'String');
            numFrames = str2num(numFrames);
            frameCount = 0;
            while(frameCount <= (numFrames))
                frameCount = frameCount + 1;
                disp(frameCount);
                preview(vid, hImage);                                                  % begin the webcam preview
                
                I = getsnapshot(vid);
                addpath('Features');
                load('SVMModel.mat');
                faceDetector = vision.CascadeObjectDetector();
                try
                    if size(I,3)==3 
                        I=rgb2gray(I);
                    end
                    
                    H = fspecial('gaussian', [3 3], .5);
                    I = imfilter(I,H,'replicate');
                    
                    [r,c]=size(I);
                    
                    bboxf = faceDetector.step(I);
                    
                    if isempty(bboxf)
                        disp('no face in the frame')
                        msgBox = msgbox({'No face in the frame'},'ERROR','custom',wrongIcon);
                        continue
                    end
                    
                    bboxf=bboxf(1,:);
                    k=imcrop(I,bboxf);
                    k=imresize(k,[96,96]);
                    k=histeq(k);
                    
                    [h1,w1]=size(k);
                    face1=k;
                    
                    %Intial right eye crop
                    x=round(w1/8);y=round( h1/6);w= round(w1/2.5);h= round(h1/2.5);
                    Ebox1=[x y w h]; 
                    k1=imcrop(k,Ebox1);
                    
                    % Haar RightEye detector.
                    faceDetector = vision.CascadeObjectDetector('RightEye');
                    bboxe1 = faceDetector.step(k1);
                    if isempty(bboxe1)
                        disp('no RightEye in the frame')
                        msgBox = msgbox({'no RightEye in the frame'},'ERROR','custom',wrongIcon);
                        continue
                    end
                    bboxe1 = bboxe1(1,:);
                    Eye1x=x+bboxe1(1)+floor(bboxe1(3)/2);
                    Eye1y=y+bboxe1(2)+floor(bboxe1(4)/2.3);
                    filePath=imcrop(k1,bboxe1);
                                        
                    %Intial left eye crop
                    x=round(w1/2);y=round( h1/6);w= round(w1/2.5);h= round(h1/2.5);
                    Ebox2=[x y w h]; %Ebox
                    k1=imcrop(k,Ebox2);
                    
                    % Haar LeftEye detector.
                    faceDetector = vision.CascadeObjectDetector('LeftEye');
                    bboxe2= faceDetector.step(k1);
                    if isempty(bboxe2)
                        disp('no LeftEye in the frame')
                        msgBox = msgbox({'no left eye in the frame'},'ERROR','custom',wrongIcon);                        
                        continue
                    end
                    bboxe2=bboxe2(1,:);
                    Eye2x=x+bboxe2(1)+round(bboxe2(3)/2.3);
                    Eye2y=y+bboxe2(2)+round(bboxe2(4)/2.3);
                    filePath=imcrop(k1,bboxe2);
                    
                    %Intial nose crop
                    x=round(w1/3.5);y=round( h1/2.5);w= round(w1/2.5);h= round(h1/2.5);
                    Ebox2=[x y w h];
                    k1=imcrop(k,Ebox2);
                    
                    % Haar Nose detector.
                    faceDetector = vision.CascadeObjectDetector('Nose');
                    bboxn= faceDetector.step(k1);
                    if isempty(bboxn)
                        disp('no Nose in the frame')
                        msgBox = msgbox({'no nose in the frame'},'ERROR','custom',wrongIcon);
                        continue;
                    end
                    bboxn=bboxn(1,:);
                    nsx=x+bboxn(1)+round(bboxn(3)/2);
                    nsy=y+bboxn(2)+round(bboxn(4)/2);
                    filePath=imcrop(k1,bboxn);
                                        
                    % 4.3 Lip Corner Detection
                    
                    %1: select coarse lips ROI using face width and nose position
                    [h1,w1]=size(k);
                    x=round(w1/4.5);y=nsy+round(h1/16);w= w1-2*x;h= h1-y;
                    lpx=x;
                    lpy=y;
                    Ebox2=[x y w h];
                    k1=imcrop(k,Ebox2);
                                        
                    %2: apply Gaussian blur to the lips ROI
                    G = fspecial('gaussian', [5 5], 2);
                    bd = imfilter(k1,G,'same');
                    
                    %3: apply horizontal sobel operator for edge detection
                    h = fspecial('sobel') ;
                    sob = imfilter(bd,h,'replicate');
                    
                    %4: apply Otsu-thresholding
                    bw=im2bw(sob);
                    
                    %5: apply morphological dilation operation
                    [sr,sc]=size(bw);
                    sr1=floor(sr/15);
                    sc1=floor(sc/12);
                    se=strel('rectangle',[sr1,sc1]);
                    dil=imclose(bw,se);
                    
                    %7: remove the spurious connected components using threshold technique to the number of pixels
                    maxArea=ceil(sr*sc/50);
                    bw1= bwareaopen(dil,maxArea);
                    
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
                    
                    yn=cn(:,2);
                    ys=min(yn);
                    yi=find(yn==ys);
                    yi=yi(1);
                    cnt=cn(yi,:);
                    
                    %9: locate the left and right most positions of connected component as lip corners
                    [R,C]=size(bw);
                    pp=stats(yi).PixelIdxList;
                    [y3,x3]=ind2sub([R C],pp);
                    
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
                    
                    lpxa=lpx+xa;
                    lpya=lpy+ya;
                    lpxb=lpx+xb;
                    lpyb=lpy+yb;

                    LipX=[lpxa,lpxb-sc1];
                    LipY=[lpya+sr1,lpyb+sr1];
                    
                    % 4.4 Eyebrow Corner Detection
                    
                    %Right Eyebrow
                    %1: select coarse  Right Eyebrow ROI using face width and eye position
                    [h1,w1]=size(k);
                    x=Eye1x;y=Eye1y-round(h1/4);w=w1/2-x;h=Eye1y-y-round(h1/30);
                    Eb1x=x;
                    Eb1y=y;
                    Ebbox1=[x y w h];
                    k1=imcrop(k,Ebbox1);
                    
                    %2: apply Gaussian blur to the lips ROI
                    G = fspecial('gaussian', [3 3], 2);
                    bd = imfilter(k1,G,'same');
                   
                    %3: apply horizontal sobel operator for edge detection
                    h = fspecial('sobel') ;
                    bw1=im2bw(sob);
                    
                    [h,w]=size(bw1);
                    bw1=bwareaopen(bw1,round(h*w/100));
                    b = padarray(bw1,[0 1],'pre');
                    b2=imclearborder(b);
                                        
                    ind=find(b2==1);
                    [Y,X]=ind2sub(size(bw1),ind);
                    x=max(X);
                    y=Y(X==x);
                    y=y(1);
                    
                    EB1x=Eb1x+x-round(w1/20);
                    EB1y=Eb1y+y;
                                        
                    %Left Eyebrow
                    %1: select coarse  Right Eyebrow ROI using face width and eye position
                    [h1,w1]=size(k);
                    x=Eye2x-round(w1/4);y=Eye2y-round(h1/4);w=round((w1-x-round(w1/15))/2);h=Eye2y-y-round(h1/30);
                    Eb2x=x;
                    Eb2y=y;
                    Ebbox1=[x y w h];
                    k1=imcrop(k,Ebbox1);
                    
                    %2: apply Gaussian blur to the lips ROI
                    G = fspecial('gaussian', [3 3], 2);
                    bd = imfilter(k1,G,'same');
                    
                    %3: apply horizontal sobel operator for edge detection
                    h = fspecial('sobel') ;
                    sob = imfilter(bd,h,'replicate');
                    bw1=im2bw(sob);
                    
                    [h,w]=size(bw1);
                    bw1=bwareaopen(bw1,round(h*w/100));
                    b = padarray(bw1,[0 1],'post');
                    b2=imclearborder(b);
                                        
                    ind=find(b2==1);
                    [Y,X]=ind2sub(size(bw1),ind);
                    x=min(X);
                    y=Y(X==x);
                    y=y(1);
                    
                    EB2x=Eb2x+x;
                    EB2y=Eb2y+y;
                    
                    % 4.5 Extraction of Active Facial Patches
                    [h1,w1]=size(k);
                    wd=round(w1/9);
                    if rem(wd,2)==0
                        wd=wd+1;
                    end
                    wg=floor(wd/2);
                    
                    P18=imcrop(k,[EB1x-wg,EB1y-wg,wd,wd]);
                    P19=imcrop(k,[EB2x-wg,EB2y-wg,wd,wd]);
                    P1=imcrop(k,[LipX(1)-wg,LipY(1)-wg,wd,wd]);
                    P9=imcrop(k,[LipX(1)-wg,LipY(1)-wg+wd,wd,wd]);
                    P4=imcrop(k,[LipX(2)-wg,LipY(2)-wg,wd,wd]);
                    P11=imcrop(k,[LipX(2)-wg,LipY(2)-wg+wd,wd,wd]);
                    P10=imcrop(k,[round((LipX(2)+LipX(1))/2)-wg,round((LipY(2)+LipY(1))/2)+wg,wd,wd]);
                    P16=imcrop(k,[round(Eye2x+Eye1x)/2-wg,round(round(Eye2y+Eye1y)/2-2*wg),wd,wd]);
                    P17=imcrop(k,[round(Eye2x+Eye1x)/2-wg,round(Eye2y+Eye1y)/2-2*wg,wd,wd]);
                    P14=imcrop(k,[Eye1x-round(1.5*wg),(round(Eye1y+.3*wg)),wd,wd]);
                    P15=imcrop(k,[round(Eye2x-0.8*wg),(round(Eye2y+.3*wg)),wd,wd]);
                    P3=imcrop(k,[round(nsx+Eye1x)/2-wg,round(nsy+Eye1y)/2-2*wg,wd,wd]);
                    P6=imcrop(k,[round(nsx+Eye2x)/2-wg,round(round(nsy+Eye2y)/2-2*wg),wd,wd]);
                    P13=imcrop(k,[nsx+2*wg+wd,nsy-wg,wd,wd]);
                    P12=imcrop(k,[nsx+2*wg+wd,nsy-wg+wd,wd,wd]);
                    P7=imcrop(k,[nsx-4*wg-wd,nsy-wg,wd,wd]);
                    P8=imcrop(k,[nsx-4*wg-wd,nsy-wg+wd,wd,wd]);
                    P5=imcrop(k,[nsx+2*wg,nsy-wg,wd,wd]);
                    P2=imcrop(k,[nsx-4*wg,nsy-wg,wd,wd]);                   
                    
                    % Feature extraction from all patches
                    %LBP
                    R=2;
                    
                    LP1=LBP(P1,R);
                    HP1=hist((LP1(:)),16);
                    
                    LP2=LBP(P2,R);
                    HP2=hist((LP2(:)),16);
                    
                    LP3=LBP(P3,R);
                    HP3=hist((LP3(:)),16);
                    
                    LP4=LBP(P4,R);
                    HP4=hist((LP4(:)),16);
                    
                    LP5=LBP(P5,R);
                    HP5=hist((LP5(:)),16);
                    
                    LP6=LBP(P6,R);
                    HP6=hist((LP6(:)),16);
                    
                    LP7=LBP(P7,R);
                    HP7=hist((LP7(:)),16);
                    
                    LP8=LBP(P8,R);
                    HP8=hist((LP8(:)),16);
                    
                    LP9=LBP(P9,R);
                    HP9=hist((LP9(:)),16);
                    
                    LP10=LBP(P10,R);
                    HP10=hist((LP10(:)),16);
                    
                    LP11=LBP(P11,R);
                    HP11=hist((LP11(:)),16);
                    
                    LP12=LBP(P12,R);
                    HP12=hist((LP12(:)),16);
                    
                    LP13=LBP(P13,R);
                    HP13=hist((LP13(:)),16);
                    
                    LP14=LBP(P14,R);
                    HP14=hist((LP14(:)),16);
                    
                    LP15=LBP(P15,R);
                    HP15=hist((LP15(:)),16);
                    
                    LP16=LBP(P16,R);
                    HP16=hist((LP16(:)),16);
                    
                    LP17=LBP(P17,R);
                    HP17=hist((LP17(:)),16);
                    
                    LP18=LBP(P18,R);
                    HP18=hist((LP18(:)),16);
                    
                    LP19=LBP(P19,R);
                    HP19=hist((LP19(:)),16);                   
                    
                    %Rotation invariant LBP_u2
                    LPr1=LBP_u2(P1,R);
                    HPr1=hist((LPr1(:)),10);
                    
                    LPr2=LBP_u2(P2,R);
                    HPr2=hist((LPr2(:)),10);
                    
                    LPr3=LBP_u2(P3,R);
                    HPr3=hist((LPr3(:)),10);
                    
                    LPr4=LBP_u2(P4,R);
                    HPr4=hist((LPr4(:)),10);
                    
                    LPr5=LBP_u2(P5,R);
                    HPr5=hist((LPr5(:)),10);
                    
                    LPr6=LBP_u2(P6,R);
                    HPr6=hist((LPr6(:)),10);
                    
                    LPr7=LBP_u2(P7,R);
                    HPr7=hist((LPr7(:)),10);
                    
                    LPr8=LBP_u2(P8,R);
                    HPr8=hist((LPr8(:)),10);
                    
                    LPr9=LBP_u2(P9,R);
                    HPr9=hist((LPr9(:)),10);
                    
                    LPr10=LBP_u2(P10,R);
                    HPr10=hist((LPr10(:)),10);
                    
                    LPr11=LBP_u2(P11,R);
                    HPr11=hist((LPr11(:)),10);
                    
                    LPr12=LBP_u2(P12,R);
                    HPr12=hist((LPr12(:)),10);
                    
                    LPr13=LBP_u2(P13,R);
                    HPr13=hist((LPr13(:)),10);
                    
                    LPr14=LBP_u2(P14,R);
                    HPr14=hist((LPr14(:)),10);
                    
                    LPr15=LBP_u2(P15,R);
                    HPr15=hist((LPr15(:)),10);
                    
                    LPr16=LBP_u2(P16,R);
                    HPr16=hist((LPr16(:)),10);
                    
                    LPr17=LBP_u2(P17,R);
                    HPr17=hist((LPr17(:)),10);
                    
                    LPr18=LBP_u2(P18,R);
                    HPr18=hist((LPr18(:)),10);
                    
                    LPr19=LBP_u2(P19,R);
                    HPr19=hist((LPr19(:)),10);
                    
                    % Features for each patch
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
                    
                    fe = [FP1,FP2,FP3,FP4,FP5,FP6,FP7,FP8,FP9,FP10,FP11,FP12,FP13,FP14,FP15,FP16,FP17,FP18,FP19];
                catch
                end
                predict = svm.predict(ModelHappy,fe);
                musicIcon = imread('Music.png');
                if predict==0
                    set(handles.axes5,'visible','on')
                    imshow(musicIcon,'parent',handles.axes5);
                    disp('Happy');
                    set(handles.edit11,'string','Happy');
                    winopen('Happy.mp3');
                else
                    predict = svm.predict(ModelSad,fe);
                    if predict==0
                        set(handles.axes5,'visible','on')
                        imshow(musicIcon,'parent',handles.axes5);
                        disp('Sad');
                        set(handles.edit11,'string','Sad');
                        winopen('Sad.mp3');    
                    else if predict==1
                            set(handles.axes5,'visible','on')
                            imshow(musicIcon,'parent',handles.axes5);
                            disp('Angry');set(handles.edit11,'string','Angry');
                            winopen('Angry.mp3');
                        end
                    end
                end
                
                if frameCount==numFrames
                    stop(vid);
                    delete(vid);
                    break;
                end
                set(handles.axes5,'visible','off')                
            end    
        else
            msgBox = msgbox({'Invalid Selection' 'You can select correct option at a time'},'ERROR','custom',wrongIcon);
        end
    end
end


function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
