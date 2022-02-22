from paramiko import SSHClient, AutoAddPolicy

ssh = SSHClient()
# 允许连接不在know_hosts文件里的主机
ssh.set_missing_host_key_policy(AutoAddPolicy())
# 连接服务器
ssh.connect(hostname='10.193.218.57', port=5000, username='yang', password='123456')
# 执行命令
commands = 'nvidia-smi'
stdin, stdout, stderr = ssh.exec_command(commands)
# 获取命令结果
res, err = stdout.read(), stderr.read()
result = res if res else err
# 将字节类型 转换为 字符串类型
result = str(result, encoding='utf-8')
print(result)
# 从远程通过ftp下载文件到本地
sftp = ssh.open_sftp()
sftp.get(remotepath='/home/yang/SAR/pkl/net_epoch_199-DenseBiasSegstac1201-Network.pkl',  localpath=r'E:\D\4dregression\voxel_upload\pkl\DenseBiassta\net_epoch_199-DenseBiasSegstac1201-Network.pkl')
commands2='ls-a'
# 关闭连接
ssh.close()
