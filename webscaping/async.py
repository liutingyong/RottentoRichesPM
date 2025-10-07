import asyncio

async def random_task(n):
    await asyncio.sleep(n)
    print(f"Task {n} completed")

async def main():
    await asyncio.gather(
        random_task(1),
        random_task(2),
        random_task(3)
    )

asyncio.run(main())