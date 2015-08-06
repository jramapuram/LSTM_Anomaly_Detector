#!/usr/local/bin/fish
set inputs 1 1 1 1 1 1
set hidden 512 256 128 64 32 16

for j in (seq 20)
		for i in (seq 6)
			python lstm_anomaly_detector.py csv --input ~/Downloads/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/A4Benchmark-TS6.csv --input_col='value' --activation=tanh --initialization=glorot_uniform --input_dim=$inputs[$i] --hidden_dim=$hidden[$i] --model_type=lstm
			python lstm_anomaly_detector.py csv --input ~/Downloads/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/A4Benchmark-TS6.csv --input_col='value' --activation=tanh --initialization=glorot_uniform --input_dim=$inputs[$i] --hidden_dim=$hidden[$i] --model_type=classical
		end
#		cp -r [* single_run
end