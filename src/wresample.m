function [ resampled, wr ] = wresample(original , ro_speed, Ts, dwr)
	%{
	Args:
		original: A vector, original time domain signal.
		roSpeed: A vector, which specify the rotational speed at each point.
		Ts: A scalar, the original sample interval(in time domain).
		dwr: The expected sample interval(in angle domain).

	%}
	N = length(original);
	w = zeros(1,N);
	w(1) = ro_speed(1) * Ts;
	for ii = 2:N
		w(ii) = w(ii-1) + ro_speed(ii)*Ts;
	end
	dwr = 10*pi*Ts;
	wr = w(1):dwr:(N-1)*dwr;
	resampled = spline(w, original, wr);

end

