
echo "Current directory: "
pwd
echo "\n"

message=$1
branch=$2

if [ -z "$message" ]; then
  message="syncing"
fi
if [ -z "$branch" ]; then
  branch="main"
fi

echo "Adding, commiting, pulling and pushing to GitHub with message: $message"

git add --all --verbose;
git commit -m "$message";
git pull --no-edit origin $branch;
git push origin $branch;