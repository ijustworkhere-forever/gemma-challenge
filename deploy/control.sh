#!/bin/bash

case "$1" in

  start)
    sudo systemctl start gemma-controller
    sudo systemctl start gemma-swarm
    sudo systemctl start gemma-leaderboard
    sudo systemctl start gemma-dashboard
    ;;

  stop)
    sudo systemctl stop gemma-controller
    sudo systemctl stop gemma-swarm
    sudo systemctl stop gemma-leaderboard
    sudo systemctl stop gemma-dashboard
    ;;

  restart)
    $0 stop
    $0 start
    ;;

  status)
    sudo systemctl status gemma-controller
    sudo systemctl status gemma-swarm
    sudo systemctl status gemma-leaderboard
    sudo systemctl status gemma-dashboard
    ;;

  *)
    echo "Usage: control.sh {start|stop|restart|status}"
    ;;
esac
