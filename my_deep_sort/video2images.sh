# https://blog.csdn.net/ternence_hsu/article/details/92980451
# https://www.cnblogs.com/jisongxie/p/9948845.html

# 30fps 持续抽 5min
fps=30
duration=300 # 5min
projectName="${fps}fps${duration}s"
inputVideo=datasets/lingshui/2021-03-06-09-52-50.mp4
outputPath="datasets/lingshui/${projectName}"
outputPattern="${outputPath}/%04d.jpg"

mkdir -p ${outputPath}
ffmpeg -i ${inputVideo} -vf fps=${fps} -t ${duration} ${outputPattern}