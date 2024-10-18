import React from 'react'
// import * as FaIcons from "react-icons/fa";
import * as AiIcons from 'react-icons/ai';
import * as IoIcons from 'react-icons/io';
import * as FaIcons from "react-icons/fa6";
import { IoStatsChartSharp } from "react-icons/io5";
import { MdOutlinePersonSearch } from "react-icons/md";
import { RiCalendarScheduleLine } from "react-icons/ri";
import { MdQuiz } from "react-icons/md";
import { RiTeamFill } from "react-icons/ri";
import { RiWechat2Line } from "react-icons/ri";
import { CiLogout } from "react-icons/ci";
export const SidebarData=[
    
    {
        title:'Home',
        path:'/home1',
        icon:<AiIcons.AiFillHome/>,
        cName:'nav-text'
    },
    {
        title:'CricBot',
        path:'/bot',
        icon:<RiWechat2Line />,
        cName:'nav-text'
    },
    {
        title:'Search Players',
        path:'/playerSearch',
        icon:<MdOutlinePersonSearch/>,
        cName:'nav-text'
    },
    {
        title:'News',
        path:'/getNews',
        icon:<FaIcons.FaRegNewspaper/>,
        cName:'nav-text'
    },
    {
        title:'Records',
        path:'/records',
        icon:<IoStatsChartSharp/>,
        cName:'nav-text'
    },
    {
        title:'Match Schedules',
        path:'/schedule',
        icon:<RiCalendarScheduleLine />,
        cName:'nav-text'
    },
    {
        title:'Players',
        path:'/card',
        icon:<RiTeamFill />,
        cName:'nav-text'
    },
    {
        title:'Play Quiz',
        path:'/quiz',
        icon:<MdQuiz />,
        cName:'nav-text'
    },
    
    {
        title:'Support',
        path:'/',
        icon:<IoIcons.IoMdHelpCircle/>,
        cName:'nav-text'
    },
    {
        title:'Logout',
        path:'/',
        icon:<CiLogout />,
        cName:'nav-text'
    },
]