import asyncio
import aiohttp
import aiofiles


async def downts(ur, na, session):
    async with session.get(ur) as resp:
        print(4)
        # with open('1.txt','w')as ff:
        #     pass
        # wirte file with async
        async with aiofiles.open('dataset/' + na, mode='wb') as f:
            await f.write(await resp.read())
            print(5)

        # async with open(f'rubb/test2/{na}','wb') as f:
        #     print(5)
        #     await f.write(await resp.content.read())

    print('over')


async def download():
    tasks = []
    async with aiohttp.ClientSession() as session:
        async with aiofiles.open('dataset/end.m3u8', mode='r', encoding='utf-8') as f:

            async for line in f:
                if str(line).startswith('#'):
                    print(1)
                    continue
                else:
                    print(2)
                    line = str(line).strip()
                    name = line.rsplit('/', 1)[1]
                    tasks.append(asyncio.create_task(downts(line, name, session)))
                await asyncio.wait(tasks)




if __name__ == '__main__':
    # progress_bar_demo()
    asyncio.run(download())
