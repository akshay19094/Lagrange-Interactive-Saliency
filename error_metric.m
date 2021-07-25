function [error_rate] = error_metric(salient_img,GT_img)
error_rate=sum( sum( salient_img ~= imbinarize(GT_img) ) ) / numel(salient_img);
           