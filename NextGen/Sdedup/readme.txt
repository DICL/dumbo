0. 이 프로그램은 Hadoop Yarn 기반으로 실행됩니다. 먼저 실행을 하시기전에 작성해야 하는 설정 파일이 있습니다.

resource_config
각 도구에서 사용할 자원과 변수를 설정할 수 있습니다.
파일 내부에 주석이 있습니다.

1. 압축을 해제합니다.
=> tar zxvf sdedup.tar.gz

2. BigBWA, Sdedup 두개의 디렉토리가 생성되며 각각의 디렉토리에서 컴파일을 해줍니다.
=> cd BigBWA;mvn package
=> cd Sdedup;mvn package

3. 실행 스크립트는 현재 2개가 있습니다. (run1.sh, run2.sh)

4. 1번 스크립트(run1.sh)의 입력은 regerence genome 파일, paired-end fastq 파일, BigBWA의 HDFS 작업 디렉토리, VCF 출력 위치 그리고 기본 설정파일 외의 설정파일을 입력 지정이 가능하고 Parallel mode 실행에 사용되는 파라미터 입력이 하나 더 있습니다.
=> run1.sh [WORK DIR(OUTPUT PATH)] [REFERNCE FILE] [FASTQ PATH (HDFS, PAIRED ONLY)] [HDFS WORK DIR] [VCF OUT DIR] (OPT. ConfigFile)

5. 2번 스크립트(run2.sh)의 입력은 regerence genome 파일, 2개의 paired-end fastq 파일, 각각의 작업 디렉토리의 위치, 설정파일, VCF출력 경로를 입력으로 실행합니다.
=> run2.sh [WORK DIR(OUTPUT PATH)] [REFERNCE FILE] [FASTQ PATH 1(HDFS, PAIRED ONLY)] [HDFS WORK DIR 1] [FASTQ PATH 2(HDFS, PAIRED ONLY)] [HDFS WORK DIR 2] [CONFIG FILE] [VCF OUT DIR]

6. 3번 스크립트(run3.sh)는 2개의 프로세스를 돌리기 위해서 각각 따로 설정파일을 작성해야합니다. 
=> run3.sh 