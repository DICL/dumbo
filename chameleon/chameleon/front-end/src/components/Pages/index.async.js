import asyncRoute from '../../lib/asyncRoute';

export const Home = asyncRoute(() => import('./Home/Home'));
export const Yarn = asyncRoute(() => import('./Yarn/Yarn'));
export const Lustre = asyncRoute(() => import('./Lustre/Lustre'));
