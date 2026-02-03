@echo off
chcp 65001 >nul
cls
echo Starting upload process...

:: 删除旧的 .git
if exist ".git" rmdir /s /q .git

:: 初始化
git init
git config user.name "ranky518"
git config user.email "ylq18525109262@163.com"

:: 创建 .gitignore
echo *.pt > .gitignore
echo *.pth >> .gitignore
echo *.csv >> .gitignore
echo in_feet/ >> .gitignore
echo users_12/ >> .gitignore
echo user_11/ >> .gitignore

:: 添加小文件
git add *.py *.md *.txt .gitignore
git commit -m "Initial commit"

:: 连接到 GitHub 并推送
git remote add origin https://github.com/ranky518/cSMART.git
git branch -M main
git push -u origin main --force

echo.
echo Done! Check https://github.com/ranky518/cSMART
pause