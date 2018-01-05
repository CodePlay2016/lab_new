function speed = pulse2speed(pulse, window_size)
	L_pulse = length(pulse);
	L_speed = L_pulse-window_size+1;
	speed = zeros(1, L_speed);
	for ii = 1:L_speed
		windowed = pulse(ii:ii+window_size-1);
		num_pulse_this = 0;
		for jj = 2:window_size
			if windowed(jj) == 1 && windowed(jj-1) == 0 
				num_pulse_this = num_pulse_this + 1;
			end
		end
		speed(ii) = num_pulse_this;
	end

