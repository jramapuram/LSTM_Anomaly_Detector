%     A timeseries is anomalous if the absolute value of the average of the latest
%     three datapoint minus the moving average is greater than three standard
%     deviations of the average. This does not exponentially weight the MA and so
%     is better for detecting anomalies with respect to the entire series.

function [anomaly] = anomaly_detector(loss, window, num_std)
    loss_mean = tsmovavg(loss, 's', window, 1);
    %loss_std = (loss - loss_mean).^2;
    anomaly = zeros(size(loss));
    for i = 2:length(loss)-1
         last_three = [loss(i-1), loss(i), loss(i+1)];
         if abs(mean(last_three) - loss_mean(i)) >= num_std * loss_mean(i)
%          if abs(loss(i) - loss_mean(i)) >= 3.0 * loss_mean(i)
            anomaly(i) = 1;
        end
    end
end