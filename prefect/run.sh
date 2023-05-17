input_val=$1

main(){
  case "${input_val}" in
    "down" ) docker-compose down ;;
    "up"   ) docker-compose down 
             docker-compose \
               --profile prefect-server \
               --profile minio \
               --profile ml-app up \
               --build -d ;;
  esac
}

main
