SELECT AVG(dep_delay), STDDEV(dep_delay), COUNT(*) FROM flights WHERE origin='ATL';
SELECT AVG(distance), STDDEV(distance), COUNT(*) FROM flights WHERE unique_carrier='TW';
SELECT unique_carrier, COUNT(*), 0, COUNT(*) FROM flights WHERE origin_state_abr='LA' GROUP BY unique_carrier;
SELECT unique_carrier, COUNT(*), 0, COUNT(*) FROM flights WHERE origin_state_abr='LA' AND  dest_state_abr='CA' GROUP BY unique_carrier;
SELECT year_date, COUNT(*), 0, COUNT(*) FROM flights WHERE origin_state_abr='LA' AND dest='JFK' GROUP BY year_date;
SELECT year_date, SUM(distance), STDDEV(distance), COUNT(*) FROM flights WHERE unique_carrier='9E' GROUP BY year_date;
SELECT origin_state_abr, SUM(air_time), STDDEV(air_time), COUNT(*) FROM flights WHERE dest='HPN' GROUP BY origin_state_abr;
SELECT unique_carrier, AVG(dep_delay), STDDEV(dep_delay), COUNT(*) FROM flights WHERE year_date=2005 AND origin='PHX' GROUP BY unique_carrier;
SELECT dest_state_abr, COUNT(*), 0, COUNT(*) FROM flights WHERE distance>2500 GROUP BY dest_state_abr;
SELECT unique_carrier, COUNT(*), 0, COUNT(*) FROM flights WHERE air_time>1000 AND dep_delay>1500 GROUP BY unique_carrier;
SELECT year_date, SUM(arr_delay*dep_delay), STDDEV(arr_delay*dep_delay), COUNT(*) FROM flights WHERE origin_state_abr = 'CA' AND dest_state_abr = 'HI' GROUP BY year_date;
SELECT dest_state_abr, SUM(taxi_out)-SUM(taxi_in), STDDEV(taxi_out-taxi_in), COUNT(*) FROM flights WHERE unique_carrier = 'UA' AND origin = 'ATL' GROUP BY dest_state_abr;