
echo "Current directory: "
pwd
echo "\n"

message=$1

if [ -z "$message" ]; then
  message="syncing"
fi

echo "Adding, commiting, pulling and pushing to GitHub with message: $message"

git add --all --verbose;
git commit -m "$message";
git pull --no-edit origin main;
git push origin main;
