import asyncio, time
from src.agents.specialized_agents import PlannerAgent
from src.agents.base_agent import InMemoryMessageBus, AgentMessage

async def main():
    bus = InMemoryMessageBus()
    planner = PlannerAgent(bus)
    msg = AgentMessage(sender='tester', receiver='planner', task_type='plan', content='Explique la maintenance prédictive de façon concise')
    try:
        t0=time.time()
        res = await planner.generate_response(msg)
        print('planner_res_len', len(res))
        print('preview:', res[:400])
        print('took', time.time()-t0)
    except Exception as e:
        print('planner_error', e)

if __name__ == '__main__':
    asyncio.run(main())
