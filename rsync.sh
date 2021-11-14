while true; 
    do
    # server
    rsync -avuP -e "ssh -p 2224" ../muzero-general suchanek@90.180.66.234:/data_l/notebooks;
    sleep 3; 
done
