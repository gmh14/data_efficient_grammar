for ((i =1; i <= $1; i++));
do
python -u retro_star_listener.py --proc_id=$i &
done
