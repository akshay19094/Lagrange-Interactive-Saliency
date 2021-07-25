clear;
clc;
close all;

% folder="Lagrange Segmentation Output - SLIC Combined - 4";
% % folder="Lagrange Segmentation Output - SLIC Oxford GrabCut - Scribble Set";
% if ~exist(folder, 'dir')
%        mkdir(folder);
% end

folder="Lagrange-Dataset\output";
if ~exist(folder, 'dir')
       mkdir(folder);
end

%Read images
path="Lagrange-Dataset\images";
warning('off');

files=dir(path);
files_count=length(files);

overall_error_rate_otsu=0;
overall_error_rate_80=0;
overall_jaccard_otsu=0;
overall_jaccard_80=0;
image_count=0;

% f=figure();

for i=3:files_count
    image_count=image_count+1;
    file=files(i).name;
    disp(file);
    if contains(file,"png")
        img=imread(path+"\"+file,"png");
    end
    if contains(file,"bmp")
        img=imread(path+"\"+file,"bmp");
        file=strrep(file,".bmp",".png");
    end
    if contains(file,"jpg")
        img=imread(path+"\"+file,"jpg");
        file=strrep(file,".jpg",".png");
    end
    
    output_file=folder+"/"+file;
%     imwrite(img,output_file);

    gt_path=strrep(path,"images","ground-truth");
    
    GT_img=imread(gt_path+"\"+file);

    scribbles_path=strrep(path,"images","scribbles-set-1");
    f=strrep(file,".png","-anno.png");
    img_scribbles_2=imread(scribbles_path+"\"+f);

    f_x=[];
    b_x=[];
    f_y=[];
    b_y=[];

    [r,c]=find(img_scribbles_2==1);
    f_x=[f_x',c']';
    f_y=[f_y',r']';
    
    [r,c]=find(img_scribbles_2==2);
    b_x=[b_x',c']';
    b_y=[b_y',r']';

    I=double(img);
    I1=I(:,:,1);
    I2=I(:,:,2);
    I3=I(:,:,3);
    
    [P,Q,~]=size(img);
    
    f_s=size(f_x,1);
    b_s=size(b_x,1);
    final_color_distance=0;
    
    rmax=0;gmax=0;bmax=0;
    fxmax=max(f_x);fymax=max(f_y);
    bxmax=max(b_x);bymax=max(b_y);
    
    %Compute color distance
    
    for k=1:f_s
        for l=1:b_s
            if k~=l
                rk=double(img(f_y(k),f_x(k),1));
                rl=double(img(b_y(l),b_x(l),1));
                gk=double(img(f_y(k),f_x(k),2));
                gl=double(img(b_y(l),b_x(l),2));
                bk=double(img(f_y(k),f_x(k),3));
                bl=double(img(b_y(l),b_x(l),3));
                color_distance=round(sqrt((rk-rl)^2+(gk-gl)^2+(bk-bl)^2));
                final_color_distance=final_color_distance+(color_distance/(f_s*b_s));
            end
        end
    end
    
    final_color_distance=final_color_distance/(sqrt((255^2)*3));
    
    disp(round(final_color_distance,2));
    
     
    color_distance_limit=0.25;
    color_distance_limit_2=0.5;
   
    %add additional background seeds
    
    if final_color_distance<color_distance_limit_2
        z=5;
        by=[z,z,P-z,P-z,round(P/4),round(P/2),round(3*P/4),round(P/4),round(P/2),round(3*P/4),z,z,z,P-z,P-z,P-z];
        bx=[z,Q-z,z,Q-z,z,z,z,Q-z,Q-z,Q-z,round(Q/4),round(Q/2),round(3*Q/4),round(Q/4),round(Q/2),round(3*Q/4)];
        b_x=[b_x',bx]';
        b_y=[b_y',by]';  
    end

    %Choosing number of superpixels
    
    if final_color_distance<color_distance_limit
        [L,NumLabels] = superpixels(img,150,'Method','slic');
    else
        if final_color_distance<color_distance_limit_2
            [L,NumLabels] = superpixels(img,125);
        else
            [L,NumLabels] = superpixels(img,300);
        end
    end
        
    %Fetch all seeds from each of the mapped superpixel region
    
    f_s=size(f_y,1);
    Xval=zeros(size(f_s,1));

    for j=1:f_s
       Xval(j)=L(round(f_y(j)),round(f_x(j)));
    end

    b_s=size(b_y,1);
    Yval=zeros(size(b_s,1));
    for j=1:b_s
       Yval(j)=L(round(b_y(j)),round(b_x(j)));
    end

    Xval=unique(Xval);
    Yval=unique(Yval);

    K=1;
    
    %Superpixels seeds
    
    error_rate_otsu_sum=0;
    error_rate_80_sum=0;
    
    jaccard_otsu_sum=0;
    jaccard_80_sum=0;
    
    prev_otsu=0;
    prev_80=0;
    
    best_otsu=0;
    best_80=0;
    
    t=1;
    
    fig=figure();
    imshow(img);
    hold on;
    plot(b_x, b_y, 'r*', 'LineWidth', 1, 'MarkerSize', 1, 'color', 'b');
    hold on;
    plot(f_x, f_y, 'r*', 'LineWidth', 1, 'MarkerSize', 1, 'color', 'r');
    hold off;
    F = getframe ;
    imwrite(F.cdata,"annotated.png");
    close;
    
    f=figure();
    f.Name = file;
    
    ax = subplot(2, 2, 1);
    imshow(img,"Parent",ax),title("Original Image");
    
    ax = subplot(2, 2, 2);
    annotated_img=imread("annotated.png");
    imshow(annotated_img,"Parent",ax),title("Annotated Image");
    
%     ax = subplot(5, 4, (4*(image_count-1))+1);
%     annotated_img=imread("annotated.png");
%     imshow(annotated_img,"Parent",ax),title("Annotated Image");
    
%     ax = subplot(2, 2, 2);
%     imshow(GT_img,"Parent",ax),title("Ground Truth");
    
    for index=1:t
        f2_x=[];f2_y=[];
        for j=1:size(Xval,2)
            [r,c]=find(L==Xval(j));
            if final_color_distance<color_distance_limit
                [~,y]=kmeans(r,K);
                [~,x]=kmeans(c,K);
                f2_x=[f2_x',x']';
                f2_y=[f2_y',y']';
            else
                f2_x=[f2_x',c']';
                f2_y=[f2_y',r']';
            end
        end

        b2_x=[];b2_y=[];
        for j=1:size(Yval,2)
            [r,c]=find(L==Yval(j));
            if final_color_distance<color_distance_limit
                [~,y]=kmeans(r,K);
                [~,x]=kmeans(c,K);
                b2_x=[b2_x',x']';
                b2_y=[b2_y',y']';
            else
                b2_x=[b2_x',c']';
                b2_y=[b2_y',r']';
            end
        end

        N2=round(double([f2_x;b2_x]));
        M2=round(double([f2_y;b2_y]));

        M=M2;N=N2;

        br=max(size(f2_x));
        [P,Q,~]=size(I);
        N(N2<1|N2>Q|M2<1|M2>P)=[];
        M(N2<1|N2>Q|M2<1|M2>P)=[];
        list=sub2ind([P,Q],M,N);

        if final_color_distance<=color_distance_limit_2
            [r,c,~]=size(img);

            alpha=0.6;
            if final_color_distance>=color_distance_limit
                alpha=0.5;
            end

            %Normalizing spatial coordinates

            f3_x=bsxfun(@times,f2_x,(alpha*255)/hypot(c,r));
            f3_y=bsxfun(@times,f2_y,(alpha*255)/hypot(c,r));
            b3_x=bsxfun(@times,b2_x,(alpha*255)/hypot(c,r));
            b3_y=bsxfun(@times,b2_y,(alpha*255)/hypot(c,r));

            M3=round([f3_x;b3_x]);
            N3=round([f3_y;b3_y]);

            ox=[M3,N3,I1(list),I2(list),I3(list)];
            [N1,M1]=ind2sub([P,Q],1:numel(I1));
            M1=bsxfun(@times,M1,(255*alpha)/hypot(c,r));
            N1=bsxfun(@times,N1,(255*alpha)/hypot(c,r));
            tt=[M1',N1',I1(:),I2(:),I3(:)];
        else
            ox=[I1(list),I2(list),I3(list)];
            tt=[I1(:),I2(:),I3(:)];
        end

        %Setting final pixel values to be passed to Lagrange Interpolation
        %algorithm

        if final_color_distance<=color_distance_limit
            ff=ox(1:br,:);
            bb=ox(br+1:end,:);
            x=[ff;bb];
            y=[ones(size(ff,1),1);zeros(size(bb,1),1)];
        else
            K=5;
            [~,ff]=kmeans(ox(1:br,:),K);
            [~,bb]=kmeans(ox(br+1:end,:),K);
            x=[round(ff);round(bb)];
            y=[ones(K,1);zeros(K,1)];
        end

        %Lagrange interpolation algorithm

        fx=zeros(P*Q,1);
        xs=size(x,1);
        for k=1:xs
            tx=ones(P*Q,1);
            for l=1:xs
                if k~=l
                   tx=tx.*(bsxfun(@minus,tt,x(l,:))/(x(k,:)-x(l,:)));
                end
            end
            fx=fx+tx*y(k);
        end
        fx(fx>1)=1;
        fx(fx<0)=0;
        fxi=reshape(fx,[P,Q]);
        salient_img=mat2gray(fxi);
        
%         ax = subplot(5, 4, (4*(image_count-1))+2);
%         imshow(salient_img,"Parent",ax),title("Lagrange Saliency Map");

        error_rate_otsu=0;
        error_rate_80=0;


        if final_color_distance>=color_distance_limit
            [counts,~] = imhist(fxi);
            T = otsuthresh(counts);
            salient_img_otsu = imbinarize(fxi,T);
            CC = bwconncomp(salient_img_otsu);
            numOfPixels = cellfun(@numel,CC.PixelIdxList);
            [~,indexOfMax] = max(numOfPixels);
            salient_img_otsu_largest_component = zeros(size(salient_img_otsu));
            salient_img_otsu_largest_component (CC.PixelIdxList{indexOfMax}) = 1;

            A=sub2ind([P Q],f_y,f_x);
            B=find(salient_img_otsu_largest_component==1);
            %If largest component doesn't have an overlap with foreground seeds
            %provided.
            while numel(intersect(A,B))==0
                salient_img_otsu(CC.PixelIdxList{indexOfMax}) = 0;
                CC = bwconncomp(salient_img_otsu);
                numOfPixels = cellfun(@numel,CC.PixelIdxList);
                [~,indexOfMax] = max(numOfPixels);
                salient_img_otsu_largest_component = zeros(size(salient_img_otsu));
                salient_img_otsu_largest_component (CC.PixelIdxList{indexOfMax}) = 1;
                A=sub2ind([P Q],f_y,f_x);
                B=find(salient_img_otsu_largest_component==1);
            end
            error_rate_otsu=error_metric(salient_img_otsu_largest_component,GT_img);
            error_rate_otsu_sum=error_rate_otsu_sum+round(error_rate_otsu,3);

            jaccard_otsu=jaccard_metric(imbinarize(salient_img_otsu_largest_component),imbinarize(GT_img));
            jaccard_otsu_sum=jaccard_otsu_sum+round(jaccard_otsu,3);

            if ((error_rate_otsu<prev_otsu)||(index==1))
                best_otsu=salient_img_otsu_largest_component;
            end
            
            prev_otsu=error_rate_otsu;
        else
            T2=0.8;
            salient_img_final = imbinarize(fxi,T2);
            salient_img_final=imfill(double(salient_img_final),'holes');

            error_rate_80=error_metric(salient_img_final,GT_img);
            error_rate_80_sum=error_rate_80_sum+round(error_rate_80,3);

            jaccard_80=jaccard_metric(imbinarize(salient_img_final),imbinarize(GT_img));
            jaccard_80_sum=jaccard_80_sum+round(jaccard_80,3);

            if ((error_rate_80<prev_80)||(index==1))
                best_80=salient_img_final;
            end
            
            prev_80=error_rate_80;
        end
    end
    
    if final_color_distance>=color_distance_limit
        error_rate=round(error_rate_otsu_sum/t,3);
        jaccard=round(jaccard_otsu_sum/t,3);
        overall_error_rate_otsu=overall_error_rate_otsu+error_rate;
        overall_jaccard_otsu=overall_jaccard_otsu+jaccard;
        k="Error rate~"+round(error_rate,3)+"["+final_color_distance+"] Jaccard~"+round(jaccard,3);
        ax = subplot(2, 2, 3);
        imshow(best_otsu,"Parent",ax),title("Binary Map");
        
        segmentedImage = img;
        segmentedImage(repmat(~best_otsu,[1 1 3])) = 0;
        redChannel = segmentedImage(:, :, 1);
        greenChannel = segmentedImage(:, :, 2);
        blueChannel = segmentedImage(:, :, 3);
        blackMask = redChannel == 0 & greenChannel == 0 & blueChannel == 0;
        redChannel(blackMask) = 0;
        greenChannel(blackMask) = 0;
        blueChannel(blackMask) = 255;
        segmentedImage = cat(3, redChannel, greenChannel, blueChannel);
        ax = subplot(2, 2, 4);
        imshow(segmentedImage,"Parent",ax),title("Segmented Image");
        
        disp(error_rate);
        disp(jaccard);
    else
        error_rate=round(error_rate_80_sum/t,3);
        jaccard=round(jaccard_80_sum/t,3);
        overall_error_rate_80=overall_error_rate_80+error_rate;
        overall_jaccard_80=overall_jaccard_80+jaccard;
        k="Error rate~"+round(error_rate,3)+"["+final_color_distance+"] Jaccard~"+round(jaccard,3);
        ax = subplot(2, 2, 3);
        imshow(best_80,"Parent",ax),title("Binary Map");
         
        segmentedImage = img;
        segmentedImage(repmat(~best_80,[1 1 3])) = 0;
        redChannel = segmentedImage(:, :, 1);
        greenChannel = segmentedImage(:, :, 2);
        blueChannel = segmentedImage(:, :, 3);
        blackMask = redChannel == 0 & greenChannel == 0 & blueChannel == 0;
        redChannel(blackMask) = 0;
        greenChannel(blackMask) = 0;
        blueChannel(blackMask) = 255;
        segmentedImage = cat(3, redChannel, greenChannel, blueChannel);
        ax = subplot(2, 2, 4);
        imshow(segmentedImage,"Parent",ax),title("Segmented Image");
        
        disp(error_rate);
        disp(jaccard);
    end
    
    output_file=strrep(output_file,".png","");
    saliency_map=strrep(output_file,".jpg","")+"_output.jpg";
    saveas(f,saliency_map);
        
%     output_file=strrep(output_file,".png","");
%     saliency_map=strrep(output_file,".jpg","")+"_saliency_map.jpg";
%     saveas(f,saliency_map);
    disp("");
end
overall_error_rate=overall_error_rate_otsu+overall_error_rate_80;
overall_jaccard=overall_jaccard_otsu+overall_jaccard_80;
disp("Overall Error Rate");
disp(overall_error_rate/image_count);
disp("Overall Jaccard Coefficient");
disp(overall_jaccard/image_count);
close all;

