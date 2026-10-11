from langchain_community.tools import (
    DuckDuckGoSearchRun,
    OpenWeatherMapQueryRun,
)

# Inbuild tools
search_tool = DuckDuckGoSearchRun()
weather_tool = OpenWeatherMapQueryRun()

# Custom tools
tools_list = [search_tool, weather_tool]
